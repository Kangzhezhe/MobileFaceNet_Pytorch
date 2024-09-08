import numpy as np
import imageio
import os
import torch
import cv2
import numpy as np
import random
import time
import random
import torchvision.transforms as transforms
from PIL import Image

def rgb2hsv_u8(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / 255.0

def hsv2rgb_u8(img):
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def enhance_image_using_hsv(img,threth = 0.1):
    hsv_img = rgb2hsv_u8(img)
    v_channel = hsv_img[..., 2]
    
    # 计算直方图和CDF
    hist, bins = np.histogram(v_channel, bins=256, range=[0, 1])
    cdf = hist.cumsum() / hist.sum()
    
    # threth = random.uniform(0.05, 0.15)
    
    min_gray_level = bins[np.searchsorted(cdf, threth)]
    max_gray_level = bins[np.searchsorted(cdf, 1.0-threth)]
    
    # 应用直方图均衡化
    v_equalized = np.clip((v_channel - min_gray_level) / (max_gray_level - min_gray_level), 0, 1)
    hsv_img[..., 2] = v_equalized
    
    return hsv2rgb_u8(hsv_img)

def random_shrink_and_move(roi_img, min_scale=0.1, max_scale=0.28, shift_range=0.1):
    h, w, _ = roi_img.shape
    mean_color = roi_img.mean(axis=(0, 1), dtype=int)


    scale = 1 - min_scale - (random.random() * (max_scale - min_scale))  # 随机缩小比例
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the ROI
    resized_roi = cv2.resize(roi_img, (new_w, new_h))

    # Calculate random shifts within the shift range
    x_shift = int((random.random() - 0.5) * 2 * w * shift_range)
    y_shift = int((random.random() - 0.5) * 2 * h * shift_range)

    # Create a new image with the same size as the original ROI, filled with zeros (black)
    new_img = np.full_like(roi_img, mean_color)


    # Calculate the center position of the original image
    x_center = w // 2
    y_center = h // 2

    # Calculate the top-left corner of the resized ROI to place it centered with the shift
    x_start = x_center - new_w // 2 + x_shift
    y_start = y_center - new_h // 2 + y_shift

    # Ensure the ROI does not exceed the boundaries of new_img
    x_end = min(w, x_start + new_w)
    y_end = min(h, y_start + new_h)
    
    # Adjust start position if end position exceeds image boundaries
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    # Ensure resized_roi fits within new_img
    resized_roi = resized_roi[:y_end - y_start, :x_end - x_start]

    new_img[y_start:y_end, x_start:x_end] = resized_roi
    
    return new_img
class MY_Face(object):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'label.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        self.transform  = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=(0.4,0.9), contrast=0.2, saturation=(0.2,0.5), hue=(-0.2,0.2)),
                transforms.RandomHorizontalFlip(),
            ], p=0.5),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.v2.imread(img_path)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)


        # augmented_img = random_shrink_and_move(img)
        # augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
        # enhanced_img = enhance_image_using_hsv(augmented_img,random.uniform(0.05, 0.15))
        # # cv2.imshow('enhanced_img',enhanced_img)
        # # enhanced_img = Image.fromarray(enhanced_img) 
        # # enhanced_img =np.array(self.transform(enhanced_img))
        # # enhanced_img = enhanced_img.transpose((1,2,0)) 
        # # cv2.imshow('enhanced_img_transform',enhanced_img)
        # # cv2.waitKey(0)
        # enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        # img = np.array(enhanced_img)
        
        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    data_dir = 'data/output'
    dataset = MY_Face(root=data_dir)
   
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for i in range(len(dataset)):
        dataset.__getitem__(i)
