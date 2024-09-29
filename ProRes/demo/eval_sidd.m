close all;clear all;

% 5_282
% 288_259
% 215_127
% 243_254
% 263_83
% 221_178
% 104_111
% 3_263
% 300_46
% 314_219
% 147_261
% 275_17
% 100_172
% 125_164
% 311_45
% 27_280

denoised = load('/sharefs/wwen/unified-vp/uip/models_inference/new3_all_lr5e-4/sidd_inference_epoch14_27_280/Idenoised.mat');
gt = load('/sharefs/wwen/unified-vp/uip/data/low_level/denoising/sidd/val/ValidationGtBlocksSrgb.mat');

denoised = denoised.Idenoised;
gt = gt.ValidationGtBlocksSrgb;
gt = im2single(gt);

total_psnr = 0;
total_ssim = 0;
for i = 1:40
    for k = 1:32
       denoised_patch = squeeze(denoised(i,k,:,:,:));
       gt_patch = squeeze(gt(i,k,:,:,:));
       ssim_val = ssim(denoised_patch, gt_patch);
       psnr_val = psnr(denoised_patch, gt_patch);
       total_ssim = total_ssim + ssim_val;
       total_psnr = total_psnr + psnr_val;
   end
end
qm_psnr = total_psnr / (40*32);
qm_ssim = total_ssim / (40*32);

fprintf('PSNR: %f SSIM: %f\n', qm_psnr, qm_ssim);