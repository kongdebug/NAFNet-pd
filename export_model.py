from networks.NAFNet_arch_export import NAFNet

import os
import argparse

import paddle

parser = argparse.ArgumentParser(description="NAFNet_export")
parser.add_argument("--save-inference-dir", type=str, default="./inference_output", help='path of model for export')
parser.add_argument("--model-dir", type=str, default="model_best.pdparams", help='path of model checkpoint')

opt = parser.parse_args()

def main(opt):

    img_channel = 3
    width = 64
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    ckpt = paddle.load(opt.model_dir)
    model.set_state_dict(ckpt['state_dict'])

    print('Loaded trained params of model successfully.')

    shape = [-1, 3, 256, 256]

    new_model = model

    new_model.eval()
    new_net = paddle.jit.to_static(
        new_model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(opt.save_inference_dir, 'model')
    paddle.jit.save(new_net, save_path)


    print(f'Model is saved in {opt.save_inference_dir}.')


if __name__ == '__main__':
    main(opt)
