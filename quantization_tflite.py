import os
import numpy as np
from onnx_tf.backend import prepare
import tensorflow as tf
from PIL import Image
import cv2

def representative_dataset_gen():
    images_folder = "data/lfw/lfw-112X96"
    
    for image_name in images_folder:
        image_path = os.path.join(images_folder,image_name)
        image = cv2.imread(image_path)
        image = image.astype(np.float32)
        image = (image - 127.5) / 128.0
        
        nhwc_data = np.expand_dims(image, axis=0)
        yield [nhwc_data]


saved_model_dir = "model/mobilefacenet.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_tflite_model = converter.convert()

with open("model/mobilefacenet_quantized.tflite", "wb") as f:
    f.write(quantized_tflite_model)