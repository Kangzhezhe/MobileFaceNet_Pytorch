import sys
# import caffe
import os
import numpy as np
# import cv2
import scipy.io
import copy

import tensorflow as tf
import core.model
import os
import torch.utils.data
from core import model
from dataloader.LFW_loader import LFW
from config import LFW_DATA_DIR
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = 'lfw-112X96'
    # folder_name = 'enhanced'
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, folds, flags]

def parseList1(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    folder_name = ''
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], 'output_' + '{}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], 'output_' + '{}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], 'output_' + '{}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], 'output_' +  '{}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, folds, flags]


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs

def getFeatureFromTFLite(lfw_dir, feature_save_dir, tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    # if os.path.exists('data/lfw_loader.pkl'):
    #     with open('data/lfw_loader.pkl', 'rb') as f:
    #         lfw_loader = pickle.load(f)
    # else:
    nl, nr, flods, flags = parseList(lfw_dir)
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                            shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    def process_data(data):
        nonlocal interpreter, input_details, output_details
        data = [d.numpy() for d in data]
        res = []

        for d in data:
            d = d.transpose(0, 2, 3, 1)
            d = (d / 0.0078125  - 1)
            d = d.astype(np.int8)
            interpreter.set_tensor(input_details[0]['index'], d)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            out = out.astype(np.float32)
            out = 0.014472329057753086  * (out - 9)
            res.append(out)

        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        return featureL, featureR

    with ThreadPoolExecutor(max_workers=8) as executor:  # 可以调整max_workers的数量
        futures = {executor.submit(process_data, data): data for data in lfw_loader}
        for future in as_completed(futures):
            featureL, featureR = future.result()
            featureLs.append(featureL)
            featureRs.append(featureR)
            count += 1
            print('extracting deep features from the face pair {}...'.format(count))

    featureLs = np.concatenate(featureLs, axis=0)
    featureRs = np.concatenate(featureRs, axis=0)

    # result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    # scipy.io.savemat(feature_save_dir, result)


def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.MobileFacenet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    file_path = 'data/lfw_loader_torch2.pkl'
    if not os.path.exists(file_path):
        nl, nr, flods, flags = parseList(lfw_dir)
        lfw_dataset = LFW(nl, nr)
        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128,
                                              shuffle=False, num_workers=16, drop_last=False)
        with open(file_path, 'wb') as f:
            pickle.dump({'lfw_loader': lfw_loader, 'flods': flods, 'flags': flags}, f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            lfw_loader = data['lfw_loader']
            flods = data['flods']
            flags = data['flags']
    # lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128,
    #                                           shuffle=False, num_workers=16, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        count += data[0].size(0)
        print('extracing deep features from the face pair {}...'.format(count))
        res = [net(d).data.cpu().numpy()for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # featureLs.append(featureL)
        # featureRs.append(featureR)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)

from onnxruntime import InferenceSession
import pickle
def getFeatureFromONNX(lfw_dir, feature_save_dir, onnx_model_path, gpu=True):
    session = InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    nl, nr, flods, flags = parseList(lfw_dir)
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                             shuffle=False, num_workers=8, drop_last=False)
    # with open('data/lfw_loader.pkl', 'wb') as f:
    #     pickle.dump(lfw_loader, f)

    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        count += data[0].size(0)
        print('extracting deep features from the face pair {}...'.format(count))

        if gpu:
            data = [d.cuda() for d in data]

        # Convert the data to numpy arrays for ONNX Runtime
        data = [d.cpu().numpy() for d in data]

        res = [session.run(None, {input_name: d})[0] for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)

        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)


# def getFeatureFromCaffe(gpu=True):
#     if gpu:
#         caffe.set_mode_gpu()
#         caffe.set_device(0)
#     else:
#         caffe.set_mode_cpu()
#     # caffe.reset_all()
#     model = '/home/xiaocc/Documents/caffe_project/sphereface/train/code/sphereface_deploy.prototxt'
#     weights = '/home/xiaocc/Documents/caffe_project/sphereface/train/result/sphereface_model.caffemodel'
#     net = caffe.Net(model, weights, caffe.TEST)
#
#     nl, nr, flods, flags = parseList()
#
#     featureLs = []
#     featureRs = []
#     for i in range(len(nl)):
#         print('extracing deep features from the {}th face pair ...'.format(i))
#         featureL = extractDeepFeature(nl[i], net)[0]
#         featureR = extractDeepFeature(nr[i], net)[0]
#         featureLs.append(featureL)
#         featureRs.append(featureR)
#     result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
#     scipy.io.savemat('caffe_result.mat', result)
#
# def extractDeepFeature(f, net, h=112, w=96):
#     img = cv2.imread(f)
#     img = (img - 127.5) / 128
#     img = img.transpose((2, 0, 1))
#     net.blobs['data'].reshape(1, 3, h, w)
#     net.blobs['data'].data[0, ...] = img
#     res = copy.deepcopy(net.forward()['fc5'])
#     net.blobs['data'].data[0, ...] = img[:, :, ::-1]
#     res_ = copy.deepcopy(net.forward()['fc5'])
#     r = np.concatenate((res, res_), 1)
#     return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--lfw_dir', type=str, default=LFW_DATA_DIR, help='The path of lfw data')
    parser.add_argument('--resume', type=str, default='./model/best/068.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default='./result/best_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()


    # getFeatureFromCaffe()
    # getFeatureFromTorch(args.lfw_dir, args.feature_save_dir, args.resume)
    # getFeatureFromONNX(args.lfw_dir, args.feature_save_dir, args.resume)
    # getFeatureFromTFLite(args.lfw_dir, args.feature_save_dir, args.resume)
    if os.path.splitext(args.resume)[1] == '.ckpt':
        getFeatureFromTorch(args.lfw_dir, args.feature_save_dir, args.resume)
    elif os.path.splitext(args.resume)[1] == '.onnx':
        getFeatureFromONNX(args.lfw_dir, args.feature_save_dir, args.resume)
    elif os.path.splitext(args.resume)[1] == '.tflite':
        getFeatureFromTFLite(args.lfw_dir, args.feature_save_dir, args.resume)
    else:
        print("Invalid resume file extension")


    ACCs = evaluation_10_fold(args.feature_save_dir)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
