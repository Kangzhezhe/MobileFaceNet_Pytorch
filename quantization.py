import torch
import torch.onnx
from torch.autograd import Variable
from core import model  # 你的模型定义
import os
import numpy as np
from PIL import Image
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader

from dataloader.LFW_loader import LFW
from lfw_eval import parseList
from onnxsim import simplify

# 加载模型
net = model.MobileFacenet()
checkpoint = torch.load('model/best/068.ckpt')
net.load_state_dict(checkpoint['net_state_dict'])
net.eval()

dummy_input = torch.randn(1, 3, 112, 96)

# 导出模型
torch.onnx.export(net, dummy_input, "model/mobilefacenet.onnx", input_names=['input'], output_names=['output'], 
                  verbose=True,opset_version=13 ,do_constant_folding=True, export_params=True)


# 定义数据预�?�理函数
def _preprocess_images(images_folder: str, height: int, width: int):
    image_names = os.listdir(images_folder)
    batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = os.path.join(images_folder, image_name)
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - np.array([123.68, 116.78, 103.94], dtype=np.float32)
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime标准
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

# 定义数据读取器类
# class XXXDataReader(CalibrationDataReader):
#     def __init__(self, calibration_image_folder: str, model_path: str):
#         self.enum_data = None

#         # 使用推理会话获取输入形状
#         session = onnxruntime.InferenceSession(model_path, None)
#         (_, _, height, width) = session.get_inputs()[0].shape

#         # 将图像转�??为输入数�??
#         self.nhwc_data_list = _preprocess_images(calibration_image_folder, height, width)
      
#         self.input_name = session.get_inputs()[0].name
#         self.datasize = len(self.nhwc_data_list)

#     def get_next(self):
#         if self.enum_data is None:
#             self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
#         return next(self.enum_data, None)

#     def rewind(self):
#         self.enum_data = None

class LFWDataReader(CalibrationDataReader):
    def __init__(self, imgl, imgr, model_path):
        self.imgl_list = imgl
        self.imgr_list = imgr
        self.enum_data = None

        # Load LFW dataset
        self.dataset = LFW(self.imgl_list, self.imgr_list)

        # Use inference session to get input shape
        session = onnxruntime.InferenceSession(model_path, None)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.dataset)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: img[0].unsqueeze(0).numpy()} for img in self.dataset])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

# 设置校准数据集路�??
lfw_dir = "data/lfw"
# 实例化数�??读取�??

nl, nr, flods, flags = parseList(lfw_dir)
dr = LFWDataReader(nl,nr, 'model/mobilefacenet.onnx')

# 配置静态量�??
conf = StaticQuantConfig(
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True)

import subprocess
result = subprocess.run("python -m onnxruntime.quantization.preprocess --input model/mobilefacenet.onnx  --output model/mobilefacenet-infer.onnx",
                         shell=True, capture_output=True, text=True)
print(result.stdout)

result = subprocess.run("onnxsim model/mobilefacenet-infer.onnx model/my_simplified_model.onnx --overwrite-input-shape 1,3,112,96",
                         shell=True, capture_output=True, text=True)
print(result.stdout)

# 执�?�量�??
quantize('model/my_simplified_model.onnx', 'model/mobilefacenet_quantized1.onnx', conf)