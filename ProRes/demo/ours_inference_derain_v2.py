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
import models_ours


import random
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from util.metrics import calculate_psnr, calculate_ssim

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--ckpt_dir', type=str, help='dir to ckpt',
                        default='/sharefs/baaivision/xinlongwang/code/uip/models/'
                        'new3_all_lr5e-4')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='uip_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='100')
    parser.add_argument('--epoch', type=int, help='model epochs',
                        default=14)
    parser.add_argument('--input_size', type=int, help='model epochs',
                        default=448)
    parser.add_argument('--split', type=int, help='model epochs', choices=[1, 2, 3, 4],
                        default=3)
    parser.add_argument('--pred_gt', action='store_true', help='trained by using gt as gt',
                        default=False)
    parser.add_argument('--save', action='store_true', help='save predictions',
                        default=False)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_ours, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def random_add_prompts_random_scales(image, prompt, prompt_range=[8,64], scale_range=[0.2,0.3]):
    image = image
    prompt = prompt
    h, w = image.shape[0],image.shape[1]
    
    mask_image = np.ones((int(h),int(w),3))
    mask_prompt = np.zeros((int(h),int(w),3))

    ratio = 0

    while (scale_range[0] > ratio) == True or (ratio > scale_range[1])!=True:
        h_p = w_p = int(random.uniform(prompt_range[0], prompt_range[1]))
        point_h = int(random.uniform(h_p, h-h_p))
        point_w = int(random.uniform(w_p, w-w_p))

        mask_image[point_h:point_h+h_p,point_w:point_w+w_p] = 0.0
        mask_prompt[point_h:point_h+h_p,point_w:point_w+w_p] = 1.0
        prompts_token_num = np.sum(mask_prompt)
        ratio = prompts_token_num/(h*w)

    # image = image*mask_image
    # prompt = prompt*mask_prompt
    image = image + prompt

    return image

def run_one_image(img, tgt, prompt_org, size, model, out_path, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    # prompt_org = torch.tensor(prompt_org)
    # # make it a batch-like
    # prompt_org = prompt_org.unsqueeze(dim=0)
    # prompt_org = torch.einsum('nhwc->nchw', prompt_org)

    # bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    # bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    # bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    # run MAE
    loss, y = model(x.float().to(device), tgt.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, :, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]

    return output.numpy()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_dir = args.ckpt_dir
    model = args.model
    epoch = args.epoch
    prompt = args.prompt
    input_size = args.input_size
    prompt_type = 'target' if args.pred_gt else 'target_sub_input'

    ckpt_file = 'checkpoint-{}.pth'.format(epoch)
    assert ckpt_dir[-1] != "/"
    dst_dir = os.path.join('models_inference', ckpt_dir.split('/')[-1],
                           "derain_inference_epoch{}_{}".format(epoch, os.path.basename(prompt).split(".")[0]))

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


    model_mae.eval()
    # datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
    if args.split == 1:
        datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200']
    elif args.split == 2:
        datasets = ['Test2800']  # this is bottleneck in time ~20min
    elif args.split == 3:
        datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
    elif args.split == 4:
        datasets = ['Rain100H']       
    else:
        raise NotImplementedError(args.split)
    
    psnr_alldatasets = []
    ssim_alldatasets = []
    print(datasets)
    img_src_dir = "datasets/low_level/derain/test/"
    for dset in datasets:
        psnr_val_rgb = []
        ssim_val_rgb = []
        real_src_dir = os.path.join(img_src_dir, dset, 'input')
        real_dst_dir = os.path.join(dst_dir, dset)
        if not os.path.exists(real_dst_dir):
            os.makedirs(real_dst_dir)
        img_path_list = glob.glob(os.path.join(real_src_dir, "*.png")) + glob.glob(os.path.join(real_src_dir, "*.jpg"))
        for img_path in tqdm.tqdm(img_path_list):
            """ Load an image """
            img_name = os.path.basename(img_path)
            out_path = os.path.join(real_dst_dir, img_name.replace('jpg', 'png'))  # TODO: save all results as pngs
            img_org = Image.open(img_path).convert("RGB")
   

            size = img_org.size
            img = img_org.resize((input_size, input_size))
            img = np.array(img) / 255.


            img = img - imagenet_mean
            img = img / imagenet_std


            prompt_org = np.load('datasets/low_level/derain.npy')
            # prompt_rand = np.load('datasets/low_level/ssid.npy')
            # alpha = 1
            # prompt_org = (1 - alpha) * prompt_org + alpha * prompt_rand

            # img = random_add_prompts_random_scales(img,prompt_org,prompt_range=[8,64],scale_range=[0.2,0.3])
            # simple add
            img = img + prompt_org

            # load gt
            rgb_gt = Image.open(img_path.replace('input', 'target')).convert("RGB")  # irrelevant to prompt-type
            
            tgt = rgb_gt.resize((input_size, input_size))
            tgt = np.array(tgt) / 255.


            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            """### Run MAE on the image"""
            # make random mask reproducible (comment out to make it change)
            torch.manual_seed(2)

            output = run_one_image(img, tgt, prompt_org, size, model_mae, out_path, device)
    
            rgb_restored = output
    
            rgb_restored = np.clip(rgb_restored, 0, 1)



            rgb_gt = np.array(rgb_gt) / 255.

            psnr = calculate_psnr(rgb_restored*255., rgb_gt*255., 0, test_y_channel=True)
            ssim = calculate_ssim(rgb_restored*255., rgb_gt*255., 0, test_y_channel=True)
            # psnr = psnr_loss(rgb_restored, rgb_gt, data_range=1)
            # ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=1)

            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)




            if args.save:
                # utils.save_img(out_path, img_as_ubyte(rgb_restored))
                output = rgb_restored * 255
                output = Image.fromarray(output.astype(np.uint8))
                output.save(out_path)

        psnr_val_rgb = sum(psnr_val_rgb) / len(img_path_list)
        ssim_val_rgb = sum(ssim_val_rgb) / len(img_path_list)
        psnr_alldatasets.append(psnr_val_rgb)
        ssim_alldatasets.append(ssim_val_rgb)
        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))

    psnr_all = sum(psnr_alldatasets) / len(datasets)
    ssim_all = sum(ssim_alldatasets) / len(datasets)
    print("PSNR: %f, SSIM: %f " % (psnr_all, ssim_all))