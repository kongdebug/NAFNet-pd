import os
from config import Config

opt = Config('training_1card.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

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


def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    print(nranks)

    ######### Set Seeds ###########
    random.seed(42)
    np.random.seed(42)
    paddle.seed(42)

    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
    log_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'logs', session)

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
    new_lr = opt.OPTIM.LR_INITIAL
    scheduler = optim.lr.CosineAnnealingDecay(learning_rate=new_lr, T_max=opt.OPTIM.T_MAX, eta_min=1e-7)
    optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=scheduler, weight_decay=0.0,  beta1=0.9, beta2=0.9)

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        # ckpt = paddle.load('model_best.pdparams')
        ckpt = paddle.load('model_latest.pdparams')
        model.set_state_dict(ckpt['state_dict'])
        optimizer.set_state_dict(ckpt['optimizer'])
        resume_iter = ckpt['iter']
        resume_step = resume_iter // opt.TRAINING.PRINT_FREQ

    ######### Loss ###########
    criterion = PSNRLoss()

    ######### DataLoaders ###########
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = PairedImageDataset_SIDD(train_dir, is_train=True)
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=8)

    val_dataset = PairedImageDataset_SIDD(val_dir, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)


    with LogWriter(logdir=log_dir) as writer:
        step = resume_step if opt.TRAINING.RESUME else 0
        best_psnr = 0
        best_iter = 0

        eval_now = 4e4
        print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

        current_iter = resume_iter if opt.TRAINING.RESUME else 0
        total_iters = opt.OPTIM.NUM_ITERS
        
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

                if current_iter % opt.TRAINING.PRINT_FREQ == 0 and local_rank == 0:
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

