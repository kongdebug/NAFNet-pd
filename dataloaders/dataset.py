# ------------------------------------------------------------------------
# Modified from HINet (https://github.com/megvii-model/HINet)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------

import os.path as osp
from paddle.io import Dataset
from paddle.vision.transforms import normalize
from utils.dataset_utils import paired_paths_from_folder, paired_random_crop, random_augmentation
from utils.image_utils import load_img, padding, img2tensor


class PairedImageDataset_SIDD(Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.
    """

    def __init__(self, data_dir, is_train):
        super(PairedImageDataset_SIDD, self).__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        
        self.gt_folder, self.lq_folder = osp.join(self.data_dir, 'gt_crops'), osp.join(self.data_dir, 'input_crops')
        self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

    def __getitem__(self, index):
        scale = 1
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        try:
            img_gt = load_img(gt_path)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        try:
            img_lq = load_img(lq_path)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.is_train:
            gt_size = 256
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = random_augmentation(img_gt, img_lq)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        return img_lq, img_gt, lq_path, gt_path
        

    def __len__(self):
        return 32*10000 if self.is_train else len(self.paths)
