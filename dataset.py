import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import random


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def data_augmentation(label, mode=0):
    # Transpose input from (c, h, w) to (h, w, c)
    label = np.transpose(label, (1, 2, 0))
    
    if mode == 0:
        # original
        augmented_label = label
    elif mode == 1:
        # flip up and down
        augmented_label = np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        augmented_label = np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        augmented_label = np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        augmented_label = np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        augmented_label = np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        augmented_label = np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        augmented_label = np.flipud(np.rot90(label, k=3))
    else:
        raise ValueError("Invalid mode. Mode must be between 0 and 7.")
    
    # Transpose output back from (h, w, c) to (c, h, w)
    augmented_label = np.transpose(augmented_label, (2, 0, 1))
    
    return augmented_label




def random_crop(img1, img2, img3, crop_size, scale = 1):
    # Validate input shape validity
    c, h, w = img1.shape
    
    assert img2.shape == (c, h, w), "img1 and img2 must have the same shape"
    assert img3.shape == (c, h // scale, w // scale), "img3 shape must match the scaling ratio"
    assert h >= crop_size and w >= crop_size, "Crop size cannot exceed image dimensions"

    if h == crop_size:
        return img1, img2, img3
    
    # Randomly generate crop starting coordinates (based on high-resolution image)
    x_index = np.random.randint(0, h - crop_size)
    y_index = np.random.randint(0, w - crop_size)

    # Crop high-resolution images
    crop_img1 = img1[:, x_index:x_index + crop_size, y_index:y_index + crop_size]
    crop_img2 = img2[:, x_index:x_index + crop_size, y_index:y_index + crop_size]

    # Compute crop region for low-resolution image
    x_index_lr = x_index // scale
    y_index_lr = y_index // scale
    crop_size_lr = crop_size // scale

    # Crop low-resolution image
    crop_img3 = img3[:, x_index_lr:x_index_lr + crop_size_lr, y_index_lr:y_index_lr + crop_size_lr]

    return crop_img1, crop_img2, crop_img3

def center_crop(img1, img2, img3, crop_size, scale=1):
    # Validate input shape validity
    c, h, w = img1.shape
    # print(img1.shape,img3.shape,scale)
    assert img2.shape == (c, h, w), "img1 and img2 must have the same shape"
    assert img3.shape == (c, h // scale, w // scale), "img3 shape must match the scaling ratio"
    assert h >= crop_size and w >= crop_size, "Crop size cannot exceed image dimensions"

    # Compute starting coordinates for center crop (based on high-resolution image)
    x_index = (h - crop_size) // 2
    y_index = (w - crop_size) // 2

    # Crop high-resolution images
    crop_img1 = img1[:, x_index:x_index + crop_size, y_index:y_index + crop_size]
    crop_img2 = img2[:, x_index:x_index + crop_size, y_index:y_index + crop_size]

    # Compute crop region for low-resolution image
    x_index_lr = x_index // scale
    y_index_lr = y_index // scale
    crop_size_lr = crop_size // scale

    # Crop low-resolution image
    crop_img3 = img3[:, x_index_lr:x_index_lr + crop_size_lr, y_index_lr:y_index_lr + crop_size_lr]

    return crop_img1, crop_img2, crop_img3


class HSData(data.Dataset):
    def __init__(self, image_dir, scale = 2, training=False, crop_size=256):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.scale = scale
        self.crop_size = crop_size
        self.argument = False
        if training:
            self.argument = True
        

    def __getitem__(self, index):
        load_dir = self.image_files[index]
        data = sio.loadmat(load_dir)
        
        ms = np.array(data['lr'][...], dtype=np.float32)
        lms = np.array(data['sr'][...], dtype=np.float32)
        gt = np.array(data['hr'][...], dtype=np.float32)
        # print(ms.shape,lms.shape,gt.shape)
        if self.argument:
            lms, gt, ms = random_crop(lms, gt, ms, crop_size=self.crop_size, scale=self.scale)
            mode = random.randint(0, 7)
            ms = data_augmentation(ms, mode=mode)
            lms = data_augmentation(lms, mode=mode)
            gt = data_augmentation(gt, mode=mode)
        else:
            lms, gt, ms = center_crop(lms, gt, ms, crop_size=256, scale=self.scale)
            
        
        ms = torch.from_numpy(ms.copy())
        lms = torch.from_numpy(lms.copy())
        gt = torch.from_numpy(gt.copy())
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)