import os
import argparse
from tqdm import tqdm
from skimage import img_as_ubyte

import paddle
from paddle.io import DataLoader

from dataloaders.dataset import PairedImageDataset_SIDD
from networks.NAFNet_arch import NAFNet
import utils


parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='./SIDD_Data/val/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/sidd/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./model_latest.pdparams',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=8, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

# config device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
paddle.set_device('gpu:0')

utils.mkdir(args.result_dir)

# load pretrained model
img_channel = 3
width = 64
enc_blks = [2, 2, 4, 8]
middle_blk_num = 12
dec_blks = [2, 2, 2, 2]

model_restoration = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

ckpt = paddle.load(args.weights)
model_restoration.set_state_dict(ckpt['state_dict'])
model_restoration.eval()

# construct dataset
test_dataset = PairedImageDataset_SIDD(args.input_dir, is_train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, drop_last=False)

# run evaluation
print("Evaluation Start")
with paddle.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0]
        rgb_gt = data_test[1]
        gt_path = data_test[-1]
        
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = paddle.clip(rgb_restored, 0, 1)

        tmp_psnr = utils.batch_PSNR(rgb_restored, rgb_gt, 1.)
        tmp_ssim = utils.batch_SSIM(rgb_restored, rgb_gt)
        
        psnr_val_rgb.append(tmp_psnr)
        ssim_val_rgb.append(tmp_ssim)

        if args.save_images:
            rgb_gt = rgb_gt.transpose([0, 2, 3, 1]).numpy()
            rgb_noisy = rgb_noisy.transpose([0, 2, 3, 1]).numpy()
            rgb_restored = rgb_restored.transpose([0, 2, 3, 1]).numpy()

            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + os.path.split(gt_path[batch])[-1], denoised_img)
print("Evaluation End")

psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)

print("PSNR: %.4f " %(psnr_val_rgb))
print("SSIM: %.4f " %(ssim_val_rgb))
