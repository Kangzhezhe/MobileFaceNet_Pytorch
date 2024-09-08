#%%
import scipy.io
import tensorflow as tf
import torch
from config import BATCH_SIZE, CASIA_DATA_DIR
from core import model  # 你的模型定义
import imageio
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from lfw_eval import parseList
import numpy as np
from numpy.linalg import norm
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# # 加载模型
# net = model.MobileFacenet()
# checkpoint = torch.load('model/best/068.ckpt')
# net.load_state_dict(checkpoint['net_state_dict'])
# net.eval()

# nl, nr, flods, flags = parseList("data/lfw")
# lfw_dataset = LFW(nl, nr)

# def compare(imgl_path ,imgr_path):

#     imgl = np.array(imageio.v2.imread(imgl_path),dtype=np.float32)
#     imgr = np.array(imageio.v2.imread(imgr_path),dtype=np.float32)

#     imgl = (imgl - 127.5) / 128.0
#     imgr = (imgr - 127.5) / 128.0

#     torch_imgl = torch.from_numpy(imgl).unsqueeze(0).permute(0, 3, 1, 2)
#     torch_imgr = torch.from_numpy(imgr).unsqueeze(0).permute(0, 3, 1, 2)
#     out_l = net(torch_imgl)
#     out_r = net(torch_imgr)

#     A = out_l.detach().numpy().flatten()
#     B = out_r.detach().numpy().flatten()

#     cosine = np.dot(A,B)/(norm(A)*norm(B))



# from onnxruntime import InferenceSession


# onnx_model_path = 'model/mobilefacenet_quantized1.onnx'  
# session = InferenceSession(onnx_model_path)
# input_name = session.get_inputs()[0].name

# def preprocess_image(image_path):
#     image = np.array(imageio.v2.imread(image_path), dtype=np.float32)
#     image = (image - 127.5) / 128.0
#     image = np.transpose(image, (2, 0, 1))  
#     image = np.expand_dims(image, axis=0)  
#     return image

# def compare(imgl_path, imgr_path):
#     imgl = preprocess_image(imgl_path)
#     imgr = preprocess_image(imgr_path)

#     out_l = session.run(None, {input_name: imgl})[0]
#     out_r = session.run(None, {input_name: imgr})[0]

#     A = out_l.flatten()
#     B = out_r.flatten()

#     cosine = np.dot(A, B) / (norm(A) * norm(B))

#     print("余弦相似:", cosine)

def read_txt(path):
    with open(path, 'r') as f:
        lines = [int(line.strip()) for line in f.readlines()]  # list 
    return lines


# imgl = preprocess_image("kkcc.jpg")

# out_l = session.run(None, {input_name: imgl})[0]
# A = out_l.flatten()
# x=np.array(read_txt("out_data1.txt"))
# B = (x+6)*0.020105

# cosine = np.dot(A, B) / (norm(A) * norm(B))
# mean_squared_error(A,B)
# print("余弦相似:", cosine)
# import ipdb;ipdb.set_trace()

# compare("kk.jpg","kk1.jpg")
# compare("kk.jpg","gjs.jpg")
# compare("kk1.jpg","gjs.jpg")

# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0002.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0003.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0004.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Abba_Eban/Abba_Eban_0001.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Alan_Zemaitis/Alan_Zemaitis_0001.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Alfredo_Pena/Alfredo_Pena_0001.jpg")
# compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Anastasia_Kelesidou/Anastasia_Kelesidou_0001.jpg")



# input = torch.randn(1, 3, 112, 96)
# output = net(input)
# import ipdb;ipdb.set_trace()

#%% ---------------------------------------------------
# pytorch 模型转换与量化
# 加载模型
model_path = 'model/best/random_crop_casia_with_my_raw_v2_20240731_210510/079.ckpt'
# model_path = 'model/best/068.ckpt'
net = model.MobileFacenet()
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net_state_dict'])
net.eval()
dummy_input = torch.randn(1, 3, 112, 96)
# 导出模型
torch.onnx.export(net, dummy_input, "model/mobilefacenet.onnx", input_names=['input'], output_names=['output'], 
                  verbose=True,opset_version=13 ,do_constant_folding=True, export_params=True)

import sys
sys.path.append("onnx2tflite")  # onnx2tflite的地址
from onnx2tflite.converter import onnx_converter
onnx_converter(
    onnx_model_path = "./model/mobilefacenet.onnx",
    need_simplify = True,
    output_path = "./model",
    target_formats = ['tflite'], #or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = True, # do quantification
    int8_mean =None, # give mean of image preprocessing 
    int8_std = None, # give std of image preprocessing 
    image_root = "data/quantization" # give image folder of train
)

import ipdb;ipdb.set_trace()

#%%
# 验证测试int8 tflite 模型
import imageio

interpreter = tf.lite.Interpreter(model_path="model/mobilefacenet_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get(img_path):
    d = imageio.imread(img_path)
    d = (d-128).astype(np.int8)
    d = np.expand_dims(d, axis=0)  
    interpreter.set_tensor(input_details[0]['index'], d)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    out = out.astype(np.int8)
    out = (out -9)
    return out.flatten()

def compare(imgl_path, imgr_path):
    imgl = get(imgl_path)
    imgr = get(imgr_path)

    A = imgl.flatten()
    B = imgr.flatten()

    cosine = np.dot(A, B) / (norm(A) * norm(B))

    print("cos:", cosine)

compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0002.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0003.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0004.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Abba_Eban/Abba_Eban_0001.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Alan_Zemaitis/Alan_Zemaitis_0001.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Alfredo_Pena/Alfredo_Pena_0001.jpg")
compare("data/lfw/lfw-112X96/Aaron_Peirsol/Aaron_Peirsol_0001.jpg","data/lfw/lfw-112X96/Anastasia_Kelesidou/Anastasia_Kelesidou_0001.jpg")
compare("debug/kk.jpg","debug/kk1.jpg")
compare("debug/kk.jpg","debug/kkcc.jpg")
compare("debug/kk1.jpg","debug/kkcc.jpg")
compare("debug/kk.jpg","debug/gjs.jpg")
compare("debug/kk1.jpg","debug/gjs.jpg")
compare("debug/kkcc.jpg","debug/gjs.jpg")
compare("data/lfw/lfw-112X96/Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg","debug/kk.jpg")
compare("data/lfw/lfw-112X96/Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg","debug/gjs.jpg")



#------------------------------------------
#%%
# import numpy as np
# import imageio
# dir = 'D:/linux/TESTAI4/MDK-ARM/'
# with open(dir+'in_data.txt', 'r') as file:
#     data = file.read().strip().split()

# data = np.array(data, dtype=np.int8)
# data = data+128
# data = data.astype(np.uint8)
# height = 112 
# width = 96 
# channels = 3  
# data = data.reshape((height, width, channels))
# imageio.imwrite('debug/compare_in_data.png', data)
# with open(dir+'out_data1.txt', 'r') as file:
#     data = file.read().strip().split()
# data = np.array(data, dtype=np.int8)
# imgl = get('debug/compare_in_data.png')
# imgr = data
# A = imgl.flatten()
# B = imgr.flatten()
# cosine = np.dot(A, B) / (norm(A) * norm(B))
# import ipdb; ipdb.set_trace()

# ------------------------------------
#%% 
#获得cubemx验证数据
interpreter = tf.lite.Interpreter(model_path="model/mobilefacenet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get(img_path):
    d = imageio.imread(img_path)
    d = d.astype(np.float32)
    d = (d-128)/128
    d = np.expand_dims(d, axis=0)  
    interpreter.set_tensor(input_details[0]['index'], d)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    out = out/ 0.015276755206286907  +11
    out = out.astype(np.int8)
    return out.flatten()

import numpy as np
import imageio
import os

dir = "data/lfw/lfw-112X96/"
inputs = []
outputs = []
cnt = 0
for dirname in os.listdir(dir):
    folder = os.path.join(dir, dirname)
    for i in  os.listdir(folder):
        file = os.path.join(folder, i)
        input = imageio.imread(file)
        input = (input-128).astype(np.int8)
        output = get(file)
        inputs.append(input)
        outputs.append(output)
        cnt+=1
        if cnt > 200:
            break
ins = np.array(inputs)
outs = np.array(outputs)
ins = ins[:200]
outs = outs[:200]
np.save('data/input.npy',ins)
np.save('data/output.npy',outs)
# import ipdb; ipdb.set_trace()

# ---------------------------------------------
# 比较cpp与python tflite
#%%
path = 'data/output_facenet.txt'
with open(path, 'r') as f:
    data = [int(i) for i in (f.readline().strip().split())]
    data = np.array(data)
    print(data)

with open('data/out_data1.txt', 'r') as file:
    data3 = [int(d) for d in file.readlines()]
data3=np.array(data3)

def cos(A,B):
    return np.dot(A, B) / (norm(A) * norm(B))
data1 = get('data/output.jpg')
data1 = np.array(data1)
print(data1)

print(cos(data,data1))

# ------------------------------------------------
# 比较python与stm32 模型结果
#%%

import numpy as np
def cos(A,B):
    return np.dot(A, B) / (norm(A) * norm(B))
import pandas as pd
df = pd.read_csv("test.csv",encoding="utf-8")
df_array = np.array(df)

dir = 'D:/linux/TESTAI4 - ����/MDK-ARM/'
out_data = read_txt(dir+'out_data1.txt')
out_data = np.array(out_data)

inputs = np.load('data/input.npy')
outputs = np.load('data/output.npy')

input = inputs[0]
output = outputs[0]
aa = input.reshape(1,-1)
bb = output.reshape(1,-1)
out_data = out_data.reshape(1,-1)
np.savetxt('data/input0.txt', aa, delimiter=',', fmt='%d')
np.savetxt('data/output0.txt', bb, delimiter=',', fmt='%d')
np.savetxt('data/out_data.txt', out_data, delimiter=',', fmt='%d')
print(cos(out_data,bb.flatten()))

# --------------------------
#%%
# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# if __name__ == '__main__':

#     trainset = CASIA_Face(root=CASIA_DATA_DIR)
#     trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

#     mean = 0.
#     std = 0.
#     n_samples = 0.

#     for data in trainloader:
#         images, _ = data
#         batch_samples = images.size(0)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#         n_samples += batch_samples
#         if n_samples >= 1000:
#             break

#     mean /= n_samples
#     std /= n_samples

#     print(f"Mean: {mean}")
#     print(f"Std: {std}")

# %% 分割数据集
with open('data/CASIA/CASIA-WebFace-112X96.txt', 'r') as infile, open('data/CASIA/CASIA-WebFace-112X96_100000.txt', 'w') as outfile:
    for i in range(100000):
        line = infile.readline()
        if not line:
            break
        outfile.write(line)


#%%     数据增强
import numpy as np
import cv2
import os
import random
import tqdm
from tqdm import tqdm
def rgb2hsv_u8(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV) / 255.0

def hsv2rgb_u8(img):
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def enhance_image_using_hsv(img):
    hsv_img = rgb2hsv_u8(img)
    v_channel = hsv_img[..., 2]
    
    # 计算直方图和CDF
    hist, bins = np.histogram(v_channel, bins=256, range=[0, 1])
    cdf = hist.cumsum() / hist.sum()
    
    # threth = random.uniform(0.05, 0.15)
    threth = 0.1
    min_gray_level = bins[np.searchsorted(cdf, threth)]
    max_gray_level = bins[np.searchsorted(cdf, 1.0-threth)]
    
    # 应用直方图均衡化
    v_equalized = np.clip((v_channel - min_gray_level) / (max_gray_level - min_gray_level), 0, 1)
    hsv_img[..., 2] = v_equalized
    
    return hsv2rgb_u8(hsv_img)

def enhance(img_path,output_path):
    img = cv2.imread(img_path)  # 读取输入图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    enhanced_img = enhance_image_using_hsv(img)
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)  # 转换回BGR格式以便保存
    # 确保目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, enhanced_img)  # 保存增强后的图像
    


# data_dir = 'data/CASIA/CASIA-WebFace-112X96'
# labels = 'data/CASIA/CASIA-WebFace-112X96.txt'
# output_dir = 'data/CASIA/enhanced'
# dir = os.path.dirname(labels)
# enhance_label = os.path.join(dir , 'CASIA-WebFace-112X96_100000_enhance.txt')
# with open(labels, 'r') as infile, open(enhance_label, 'w') as outfile:
#     for line in tqdm(infile.readlines()):
#         if not line:
#             break
#         data_path = os.path.join(data_dir, line.split()[0])
#         label = int(line.split()[1])
#         out_path = os.path.join(output_dir, line.split()[0])
#         # outfile.write(out_path + ' ' + str(label) + '\n')
#         # print(out_path)
#         enhance(data_path, out_path)

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

data_dir = 'data/CASIA/CASIA-WebFace-112X96'
labels = 'data/CASIA/CASIA-WebFace-112X96.txt'
output_dir = 'data/CASIA/enhanced'
dir = os.path.dirname(labels)
enhance_label = os.path.join(dir , 'CASIA-WebFace-112X96_100000_enhance.txt')

def process_line(line):
    if line:
        data_path = os.path.join(data_dir, line.split()[0])
        label = int(line.split()[1])
        out_path = os.path.join(output_dir, line.split()[0])
        enhance(data_path, out_path)
        return out_path + ' ' + str(label) + '\n'
    return None

with open(labels, 'r') as infile, open(enhance_label, 'w') as outfile:
    lines = infile.readlines()
    with ThreadPoolExecutor(max_workers=8) as executor:  # 你可以根据你的系统调整max_workers的数量
        future_to_line = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(future_to_line), total=len(lines)):
            result = future.result()
            if result:
                outfile.write(result)


# output_dir = 'data/lfw/enhanced'
# data_dir = 'data/lfw/lfw-112X96'
# for filedir in tqdm(os.listdir(data_dir)):
#     for filename in tqdm(os.listdir(os.path.join(data_dir, filedir))):
#         input_path = os.path.join(data_dir,filedir, filename)
#         output_path = os.path.join(output_dir,filedir, filename)
#         name = os.path.splitext(output_path)[0]
#         name_path = name + '.jpg'
#         enhance(input_path, name_path)
#         print(f'{input_path} -> {name_path}')

#%%
import os
import cv2
from tqdm import tqdm
import shutil

dir = 'data/output'

save_dir = 'data/quantization'
os.makedirs(save_dir, exist_ok=True)
cnt = 0
for filedir in os.listdir(dir):
    if not os.path.isdir(os.path.join(dir, filedir)):
        continue
    for filepath in os.listdir(os.path.join(dir, filedir)):
        file_name = os.path.join(dir, filedir, filepath)
        name = os.path.splitext(file_name)[0]
        if int(name[-1]) % 2 != 0:
            base_name = os.path.basename(file_name)
            save_name = os.path.join(save_dir,filedir+base_name)
           # print(file_name, save_name)
            shutil.copyfile(file_name, save_name)
            #print(file_name, int(name[-1]))
        cnt += 1
        if cnt >500:
            break

        # cv2.imwrite(os.path.join(dir, filedir, filepath), img)


import cv2
import numpy as np
import random
import time
import random

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

#%%
# CASIA 数据增强

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

data_dir = 'data/CASIA/CASIA-WebFace-112X96'
labels = 'data/CASIA/CASIA-WebFace-112X96.txt'
output_dir = 'data/CASIA/enhanced_resize'
os.makedirs(output_dir, exist_ok=True)
dir = os.path.dirname(labels)
enhance_label = os.path.join(dir , 'CASIA-WebFace-112X96_100000_enhance.txt')

def process_line(line):
    if line:
        data_dir = 'data/CASIA/CASIA-WebFace-112X96'
        labels = 'data/CASIA/CASIA-WebFace-112X96.txt'
        output_dir = 'data/CASIA/enhanced_resize'
        data_path = os.path.join(data_dir, line.split()[0])
        label = int(line.split()[1])
        out_path = os.path.join(output_dir, line.split()[0])
        # enhance(data_path, out_path)
        roi_img = cv2.imread(data_path)
        augmented_img = random_shrink_and_move(roi_img)
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        enhanced_img = enhance_image_using_hsv(augmented_img,random.uniform(0.05, 0.2))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

        output_dir = os.path.dirname(out_path)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(out_path, enhanced_img)
        return out_path + ' ' + str(label) + '\n'
    return None

with open(labels, 'r') as infile, open(enhance_label, 'w') as outfile:
    lines = infile.readlines()
    with ThreadPoolExecutor(max_workers=8) as executor:  # 你可以根据你的系统调整max_workers的数量
        future_to_line = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(future_to_line), total=len(lines)):
            result = future.result()
            if result:
                outfile.write(result)

def aug_img(roi_img):
    augmented_img = random_shrink_and_move(roi_img)
    augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
    enhanced_img = enhance_image_using_hsv(augmented_img,random.uniform(0.05, 0.15))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    return enhanced_img

def main():
    random.seed(int(time.time()))

    roi_img = cv2.imread('data/CASIA/CASIA-WebFace-112X96/0000045/003.jpg')  # Replace with your ROI image path

    # Generate 5 augmented images
    enhanced_img = aug_img(roi_img)
    cv2.imwrite("roitest.jpg", enhanced_img)



#%%
# 增强lfw
output_dir_1 = 'data/lfw/enhanced'
data_dir = 'data/lfw/lfw-112X96'

for filedir in tqdm(os.listdir(data_dir)):
    for filename in os.listdir(os.path.join(data_dir, filedir)):
        input_path = os.path.join(data_dir, filedir, filename)
        output_path = os.path.join(output_dir_1, filedir, filename)
        name = os.path.splitext(output_path)[0]
        name_path = name + '.jpg'
        output_dir = os.path.dirname(name_path)
        os.makedirs(output_dir, exist_ok=True)
        roi_img = cv2.imread(input_path)
        augmented_img = random_shrink_and_move(roi_img)
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        enhanced_img = enhance_image_using_hsv(augmented_img, random.uniform(0.05, 0.2))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name_path, enhanced_img)
        # print(f'{input_path} -> {name_path}')