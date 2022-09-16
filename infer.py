# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from paddle import inference
import numpy as np
import utils

from reprod_log import ReprodLogger
from skimage.metrics import peak_signal_noise_ratio

def batch_PSNR(img, imclean, data_range):
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="NAFNET DENOISING", add_help=add_help)

    parser.add_argument(
        '--clean-dir',
        type=str,
        default="SIDD_patches/val_mini/gt_crops/ValidationBlocksSrgb_0.png",
        help='path to clean data')
    parser.add_argument(
        '--noisy-dir',
        type=str,
        default="SIDD_patches/val_mini/input_crops/ValidationBlocksSrgb_0.png",
        help='path to inference data')

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=True, type=str2bool, help="use_gpu")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")

    args = parser.parse_args()
    return args


class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "model.pdmodel"),
            os.path.join(args.model_dir, "model.pdiparams"))


    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, clean_dir, noisy_dir):
        """preprocess
        Preprocess to the input.
        Args:
            clean_dir: path to clean data.
            noisy_dir: path to noisy data
        Returns: Input data after preprocess.
        """
        clean = utils.load_img(clean_dir)
        noisy = utils.load_img(noisy_dir)

        clean = clean.transpose([2, 0, 1])
        noisy = noisy.transpose([2, 0, 1])

        clean = np.expand_dims(clean, 0)
        noisy = np.expand_dims(noisy, 0)

        return clean, noisy

    def postprocess(self, restored, clean):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            restored: Inference denoised image.
            clean: Clean image
        Returns: Output denoised image.
        """
        restored = np.clip(restored, 0., 1.)
        psnr = batch_PSNR(restored, clean, 1.)

        return psnr

    def run(self, data):
        """run
        Inference process using inference engine.
        Args:
            data: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(data)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output




def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="NAFNet_denoising",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # dataset preprocess
    clean, noisy = inference_engine.preprocess(args.clean_dir,args.noisy_dir)
    if args.benchmark:
        autolog.times.stamp()

    restored = inference_engine.run(noisy)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    psnr = inference_engine.postprocess(restored, clean)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"image_name: {args.noisy_dir}, psnr: {psnr}")
    return psnr


if __name__ == "__main__":
    args = get_args()
    psnr = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("psnr", np.array([psnr]))
    reprod_logger.save("output_inference_engine.npy")
