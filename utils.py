import torch
import logging
from ssim_torch import ssim
import math


def torch_rmse(img, ref):
    """
    Computes the Root Mean Square Error (RMSE) between two tensors.
    :param img: Tensor, input image
    :param ref: Tensor, reference image
    :return: Tensor, RMSE value
    """
    squared_diff = (img - ref) ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = torch.sqrt(mean_squared_diff)
    return rmse

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ergas(img, ref, h=1):
    """
    Computes the Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) between two tensors.
    :param img: Tensor, input image
    :param ref: Tensor, reference image
    :param h: float, resolution ratio, default: 1
    :return: Tensor, ERGAS value
    """
    # Assume input shape is [channels, height, width]
    b = img.size(0)  # number of bands
    
    mse_between_images = ((img - ref) ** 2).mean(dim=(1, 2))  # Compute MSE per band
    mean_ref_per_band = ref.mean(dim=(1, 2))
    
    ergas = 100 * h * torch.sqrt((mse_between_images / (mean_ref_per_band ** 2)).mean())
    return ergas

def torch_sam(img, ref):  # input [28,256,256]
    # Compute dot product
    dot_product = (img * ref).sum(dim=0)
    
    # Compute L2 norm for each pixel
    img_norm = torch.norm(img, p=2, dim=0)
    ref_norm = torch.norm(ref, p=2, dim=0)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    cos_theta = dot_product / (img_norm * ref_norm + epsilon)
    
    # Ensure numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Compute spectral angle (in radians)
    sam_radians = torch.acos(cos_theta)
    
    # Return the mean spectral angle across all pixels
    return sam_radians.mean()*180/math.pi

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger