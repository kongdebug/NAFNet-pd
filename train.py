import os
import argparse
import paddle

import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle import nn
import random
import time
import numpy as np

import utils
from dataloaders.dataset import PairedImageDataset_SIDD

from networks.NAFNet_arch import NAFNet
from losses import PSNRLoss
import paddle.distributed as dist

from visualdl import LogWriter


parser = argparse.ArgumentParser(description="HINet_TIPC_train")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--iter", type=int, default=16, help="Number of training iterations")
parser.add_argument("--lr", type=float, default=0.000225, help="Initial learning rate")
parser.add_argument("--tmax", type=int, default=3200000, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default="SIDD_patches/train/", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="SIDD_patches/val/", help="path of val dataset")
parser.add_argument("--log_dir", type=str, default="output", help="path of save results")
parser.add_argument("--print_freq", type=int, default=2, help="Training print frequency")

opt = parser.parse_args()

def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
   
    print(nranks)

    ######### Set Seeds ###########
    random.seed(42)
    np.random.seed(42)
    paddle.seed(42)

    ######### Save Dir ###########
    result_dir = os.path.join(opt.log_dir, 'results')
    model_dir = os.path.join(opt.log_dir, 'models')
    log_dir = os.path.join(opt.log_dir, 'log')


    if local_rank == 0:
        utils.mkdir(result_dir)
        utils.mkdir(model_dir)

    ######### Model ###########
    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    model.train()

    ######### Scheduler ###########
    new_lr = opt.lr
    scheduler = optim.lr.CosineAnnealingDecay(learning_rate=new_lr, T_max=opt.tmax, eta_min=1e-7)
    optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=scheduler, weight_decay=0.0,  beta1=0.9, beta2=0.9)

    ######### Loss ###########
    criterion = PSNRLoss()

    ######### DataLoaders ###########
    train_dir = opt.data_dir
    val_dir = opt.val_dir

    train_dataset = PairedImageDataset_SIDD(train_dir, is_train=True)
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=8)

    val_dataset = PairedImageDataset_SIDD(val_dir, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)


    with LogWriter(logdir=log_dir) as writer:
        step = 0
        best_psnr = 0
        best_iter = 0
        
        current_iter = 0
        total_iters = opt.iter
        
        eval_now = 5e4 if total_iters > 1e4 else 10
        print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

        while current_iter <= total_iters:
            epoch_start_time = time.time()
            for data in train_loader:
                current_iter += 1
                if current_iter > total_iters:
                    break
                input_lq = data[0]
                gt = data[1]
                if nranks > 1:
                    outputs = ddp_model(input_lq)
                else:
                    outputs = model(input_lq)

                l_total = 0.0
                if not isinstance(outputs, list):
                    outputs = [outputs]
                for output in outputs:
                    l_total += criterion(output, gt)

                optimizer.clear_grad()
                l_total.backward()
                optimizer.step()

                if current_iter % opt.print_freq == 0 and local_rank == 0:
                    step += 1
                    writer.add_scalar(tag='loss', value=l_total.item(), step=step)
                    writer.add_scalar(tag='lr', value=optimizer.get_lr(), step=step)
                    print("Iter: {}\tTime: {:.4f}\tLoss: {:.4f}\tLR: {:.6f}".format(current_iter, time.time() - epoch_start_time, l_total.item(), optimizer.get_lr()))
                    
                if current_iter % eval_now == 0 and local_rank == 0:
                    model.eval()
                    with paddle.no_grad():
                        psnr_val_rgb = []
                        for data_val in val_loader:
                            input_lq = data_val[0]
                            gt = data_val[1]

                            output = model(input_lq)
                            output = paddle.clip(output, 0, 1)
                            psnr_val_rgb.append(utils.batch_PSNR(output, gt, 1.))

                        psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)

                        if psnr_val_rgb > best_psnr:
                            best_psnr = psnr_val_rgb
                            best_iter = current_iter
                            paddle.save({'iter': current_iter,
                                         'state_dict': model.state_dict(),
                                         'optimizer': optimizer.state_dict()
                                         }, os.path.join(model_dir, "model_best.pdparams"))

                        print(
                            "[iter %d\t PSNR SIDD: %.4f\t] ----  [best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                                current_iter, psnr_val_rgb, best_iter, best_psnr))

                    writer.add_scalar(tag='PSNR_val', value=psnr_val_rgb, step=step)

                    model.train()

                # update lr
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    lr_sche = optimizer.user_defined_optimizer._learning_rate
                else:
                    lr_sche = optimizer._learning_rate
                if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                    lr_sche.step()

            if local_rank == 0:
                paddle.save({'iter': current_iter,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }, os.path.join(model_dir, "model_latest.pdparams"))

if __name__ == '__main__':
    main()
