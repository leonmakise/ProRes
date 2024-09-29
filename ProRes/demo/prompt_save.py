# -*- coding: utf-8 -*-

import sys
import os
import warnings

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')

import models_unet
import models_mirnetv2
import models_mprnet

import random
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from collections import OrderedDict
import cv2

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--ckpt_dir', type=str, help='dir to ckpt',
                        default='/sharefs/baaivision/xinlongwang/code/uip/models/'
                        'new3_all_lr5e-4')
                        # 'new_ablation_bs2x32x4_enhance_gt_300ep_sl1_beta0.01_square896x448_fusefeat_mask0.75_merge3')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='uip_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='100')
    parser.add_argument('--epoch', type=int, help='model epochs',
                        default=14)
                        # default=150)
    parser.add_argument('--input_size', type=int, help='model epochs',
                        default=448)
    parser.add_argument('--pred_gt', action='store_true', help='trained by using gt as gt',
                        default=False)
    parser.add_argument('--save', action='store_true', help='save predictions',
                        default=False)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mprnet, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)

    # state_dict = checkpoint["state_dict"]
    # model.load_state_dict(state_dict)
    return model




def run_one_image(img, tgt, type_dict, size, model, out_path, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)



    # run MAE
    loss, y = model(x.float().to(device), tgt.float().to(device),type_dict.float().to(device))
    y = torch.einsum('nchw->nhwc', y[0]).detach().cpu()

    output = y[0, :, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]

    return output.numpy()


# TODO: modified from impl. in git@github.com:swz30/MIRNet.git
def myPSNR(tar_img, prd_img):
    imdff = np.clip(prd_img, 0, 1) - np.clip(tar_img, 0, 1)
    rmse = np.sqrt((imdff ** 2).mean())
    ps = 20 * np.log10(1 / rmse)
    return ps


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_dir = args.ckpt_dir
    model = args.model
    epoch = args.epoch
    prompt = args.prompt
    input_size = args.input_size
    prompt_type = 'gt' if args.pred_gt else 'gt_sub_input'

    ckpt_file = 'checkpoint-{}.pth'.format(epoch)
    assert ckpt_dir[-1] != "/"
    dst_dir = os.path.join('models_inference', ckpt_dir.split('/')[-1],
                           "enhance_inference_epoch{}_{}".format(epoch, os.path.basename(prompt).split(".")[0]))

    if os.path.exists(dst_dir):
        # raise Exception("{} exist! make sure to overwrite?".format(dst_dir))
        warnings.warn("{} exist! make sure to overwrite?".format(dst_dir))
    else:
        os.makedirs(dst_dir)
    print("output_dir: {}".format(dst_dir))

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    model_mae = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_mae.to(device)

    prompt = model_mae.prompt
    prompt = torch.einsum('nchw->nhwc', prompt)

    output = prompt[3, :, :, :].detach().cpu().numpy()
    output_min, output_max = output.min(), output.max()
    output_normalized = (output - output_min) / (output_max - output_min)*255
    print(output_normalized.max(), output_normalized.min())
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([output_normalized], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    plt.savefig(os.path.join('./models_inference/', str(3) + '.png'))

        # # print(output.dtype, output.shape)
        # # np.save(os.path.join('./models_inference/', str(i) + '.npy'), output)
        # # print(output.max(), output.min())
        # output = output * imagenet_std + imagenet_mean
        # print(output.max(), output.min())
        # output = output * 255
        # output_min, output_max = output.min(), output.max()
        # output_normalized = (output - output_min) / (output_max - output_min)
        # # print(output_normalized.max(), output_normalized.min())
        # output = Image.fromarray(output.astype(np.uint8))
        # output.save(os.path.join('./models_inference/', str(i) + '.png'))


    # # img_src_dir = "datasets/low_level/denoising/sidd/sidd_val_patch256/input"
    # # type_dict = torch.tensor([0, 0, 1, 0]).unsqueeze(0).unsqueeze(0).cuda()

    # # img_src_dir = "datasets/low_level/deblur/test/GoPro/input"    
    # # type_dict = torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).cuda()
    
    # # img_src_dir = "datasets/low_level/derain/test/Rain100L/input"    
    # # type_dict = torch.tensor([1, 0, 0, 0]).unsqueeze(0).unsqueeze(0).cuda()
    
    # img_src_dir = "datasets/low_level/enhance/lol/eval15/input"
    # type_dict = torch.tensor([0, 1, 0, 0]).unsqueeze(0).unsqueeze(0).cuda()

    # # img_src_dir = "datasets/low_level/deblur/train/input_crop" 
    
    # img_path_list = glob.glob(os.path.join(img_src_dir, "*.png"))



    # model_mae.eval()
    # for img_path in tqdm.tqdm(img_path_list):
    #     """ Load an image """
    #     img_name = os.path.basename(img_path)
    #     out_path = os.path.join(dst_dir, img_name)
    #     img = Image.open(img_path).convert("RGB")

    #     size = img.size
    #     img = img.resize((input_size, input_size))
    #     img = np.array(img) / 255.

    #     # load gt
    #     # rgb_gt = Image.open(img_path.replace('input', 'groundtruth')).convert("RGB")  
    #     # rgb_gt = Image.open(img_path.replace('input', 'target')).convert("RGB")  
    #     rgb_gt = Image.open(img_path.replace('input', 'gt')).convert("RGB")  


    #     # irrelevant to prompt-type
    #     rgb_gt = rgb_gt.resize((input_size, input_size))
    #     rgb_gt = np.array(rgb_gt) / 255.


    #     # normalize by ImageNet mean and std
    #     img = img - imagenet_mean
    #     img = img / imagenet_std

    #     tgt = rgb_gt  # tgt is not available
    #     # normalize by ImageNet mean and std
    #     tgt = tgt - imagenet_mean
    #     tgt = tgt / imagenet_std

    #     """### Run MAE on the image"""
    #     # make random mask reproducible (comment out to make it change)
    #     # torch.manual_seed(2)

    #     output = run_one_image(img, tgt, type_dict, size, model_mae, out_path, device)
  
    #     rgb_restored = output
 
    #     rgb_restored = np.clip(rgb_restored, 0, 1)



    #     if args.save:
    #         # utils.save_img(out_path, img_as_ubyte(rgb_restored))
    #         output = rgb_restored * 255
    #         output = Image.fromarray(output.astype(np.uint8))
    #         output.save(out_path)

