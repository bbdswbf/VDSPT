from utils import *

import os
import torch
import time
import numpy as np
import datetime
from option import opt
from loss import HybridLoss
# Data loaders
from dataset import HSData
from torch.utils.data import DataLoader
 
from VDSPT import VDSPT


device = torch.device(f"cuda:{opt.gpu_id}")



# Load datasets. Custom data loading functions should be implemented for different datasets.
train_set = HSData(opt.data_root+"_x"+str(opt.scale)+"/train", scale=opt.scale, training=True)
test_set = HSData(opt.data_root+"_x"+str(opt.scale)+"/test", scale=opt.scale)


train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1,shuffle=False)



# Set checkpoint save directory
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
model_path = opt.outf + opt.model_name  + '_dim'+str(opt.dimention) + '_stg'+str(opt.stage) + '_X' + str(opt.scale) + opt.data_name + date_time + '/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model = VDSPT(dim=opt.dimention, stage=opt.stage, num_blocks=[2, 1], in_channel=opt.in_ch, out_channel= opt.out_ch).to(device)



# Configure optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate , betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)


hy_loss = HybridLoss().to(device)


# Training function
def train(epoch, logger, train_loader, model):
    epoch_loss = 0
    begin = time.time()
    for iteration, (lr, sr, hr) in enumerate(train_loader):
        lr, sr, hr = lr.to(device), sr.to(device), hr.to(device)
        optimizer.zero_grad()
        model_out = model(sr)
        loss = hy_loss(model_out, hr)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".
                format(epoch, epoch_loss / 2, (end - begin)))
    return 0

# Test function
def test(epoch, logger, test_loader, model):
    psnr_list, ssim_list, sam_list, rmse_list, ergas_list = [], [], [], [], []
    model.eval()
    begin = time.time()

    for iteration, (lr, sr, hr) in enumerate(test_loader):
        lr, sr, hr = lr.to(device), sr.to(device), hr.to(device)
        with torch.no_grad():
            model_out = model(sr)
        
        psnr_val = torch_psnr(model_out[0, :, :, :], hr[0, :, :, :])
        ssim_val = torch_ssim(model_out[0, :, :, :], hr[0, :, :, :])
        sam_val = torch_sam(model_out[0, :, :, :], hr[0, :, :, :])
        rmse_val = torch_rmse(model_out[0, :, :, :], hr[0, :, :, :])
        ergas_val = torch_ergas(model_out[0, :, :, :], hr[0, :, :, :])

        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val.detach().cpu().numpy())
        rmse_list.append(rmse_val.detach().cpu().numpy())
        ergas_list.append(ergas_val.detach().cpu().numpy())
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))
    rmse_mean = np.mean(np.asarray(rmse_list))
    ergas_mean = np.mean(np.asarray(ergas_list))

    end = time.time()
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, rmse = {:.4f}, ergas = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,sam_mean,rmse_mean,ergas_mean,(end - begin)))
    model.train()
    return psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))

    #test(0, logger, test_loader, model)
    
    # Initialize best PSNR value and list of saved model checkpoints
    best_psnr = 0.0
    saved_models = []  # List of saved model weight filenames

    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger, train_loader, model)
        (psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger, test_loader, model)
        scheduler.step()

        # Check if PSNR exceeds the current best after 100 training epochs
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            model_name = f'model_epoch_{epoch}_psnr_{psnr_mean:.2f}_ssim_{ssim_mean:.2f}.pth'
            model_path_save = os.path.join(model_path, model_name)
            
            # Save model weights
            torch.save(model.state_dict(), model_path_save)
            logger.info(f"===> Saved new best model: {model_name} with PSNR: {psnr_mean:.2f}")
            
            # Add the newly saved model to the list
            saved_models.append(model_name)
            
            # Remove the oldest model if more than 3 checkpoints are stored
            if len(saved_models) > 3:
                oldest_model = saved_models.pop(0)
                os.remove(os.path.join(model_path, oldest_model))
                logger.info(f"===> Removed old model: {oldest_model}")

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()