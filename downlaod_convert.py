import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
from mxnet.gluon.model_zoo.vision import *                                     
import time
import os
import sys
if len(sys.argv) == 2:
    model_name = sys.argv[1]
    p = 'none'
elif len(sys.argv) == 3:
    model_path = sys.argv[1]
    p = sys.argv[2]
else:
    print('python3 download_convert.py <model_name>')
    print('or download all model and convert')
    print('python3 download_convert.py <model_name> all')
    sys.exit()
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
batch_shape = (1, 3, 244, 244)
folder_pretrain_model = 'pretrain_model'
folder_onnx_model = 'onnx_model'
if not os.path.exists(folder_pretrain_model):
    os.makedirs(folder_pretrain_model)
if not os.path.exists(folder_onnx_model):
    os.makedirs(folder_onnx_model)
if p == "all" or p == "All" or p == "ALL":
    model_list_all = ['inception_v3','squeezenet1_1','resnet18_v1','resnet34_v1','resnet50_v1','resnet101_v1','resnet152_v1','resnet18_v2','resnet34_v2','resnet50_v2','resnet101_v2','resnet152_v2','alexnet','densenet121','densenet161','densenet169','densenet201','mobilenet0_25','mobilenet0_5','mobilenet0_75','mobilenet1_0','mobilenet_v2_0_25','mobilenet_v2_0_5','mobilenet_v2_0_75','mobilenet_v2_1_0','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','vgg11','vgg13','vgg16','vgg19']
else:
    model_list_all = [model_name]
for pretrain_name in model_list_all:
    print('Download model ,save pretrain model and convert to onnx model : ',pretrain_name)
    if pretrain_name == 'resnet18_v1':
        net = resnet18_v1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet34_v1':
        net = resnet34_v1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet50_v1':
        net = resnet50_v1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet101_v1':
        net = resnet101_v1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet152_v1':
        net = resnet152_v1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet18_v2':
        net = resnet18_v2(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet34_v2':
        net = resnet34_v2(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet50_v2':
        net = resnet50_v2(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet101_v2':
        net = resnet101_v2(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'resnet152_v2':
        net = resnet152_v2(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'alexnet':
        net = alexnet(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'densenet121':
        net = densenet121(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'densenet161':
        net = densenet161(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'densenet169':
        net = densenet169(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'densenet201':
        net = densenet201(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'inception_v3':
        net = inception_v3(pretrained=True,ctx=mx.cpu())
        batch_shape = (1, 3, 299, 299)
    elif pretrain_name == 'vgg11':
        net = vgg11(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg13':
        net = vgg13(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg16':
        net = vgg16(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg19':
        net = vgg19(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg11_bn':
        net = vgg11_bn(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg13_bn':
        net = vgg13_bn(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg16_bn':
        net = vgg16_bn(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'vgg19_bn':
        net = vgg19_bn(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'squeezenet1_0':
        net = squeezenet1_0(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'squeezenet1_1':
        net = squeezenet1_1(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet0_25':
        net = mobilenet0_25(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet0_5':
        net = mobilenet0_5(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet0_75':
        net = mobilenet0_75(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet1_0':
        net = mobilenet1_0(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet_v2_0_25':
        net = mobilenet_v2_0_25(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet_v2_0_5':
        net = mobilenet_v2_0_5(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet_v2_0_75':
        net = mobilenet_v2_0_75(pretrained=True,ctx=mx.cpu())
    elif pretrain_name == 'mobilenet_v2_1_0':
        net = mobilenet_v2_1_0(pretrained=True,ctx=mx.cpu())
    else:
        print('Unknow model from pretrain ImageNet mxnet.')
    net.hybridize()
    net.forward(mx.nd.zeros(batch_shape))
    if not os.path.exists(folder_pretrain_model+"/"+pretrain_name):
        net.export(folder_pretrain_model+"/"+pretrain_name)
    else:
        print('Already have a model : ',folder_pretrain_model+"/"+pretrain_name)
    sym = folder_pretrain_model+'/'+pretrain_name+'-symbol.json'
    params = folder_pretrain_model+'/'+pretrain_name+'-0000.params'
    onnx_file = folder_onnx_model+'/'+pretrain_name+'.onnx'
    if not os.path.exists(onnx_file):
        converted_model_path = onnx_mxnet.export_model(sym, params, [batch_shape], np.float32, onnx_file)
    else:
        print('Already have a model : ',onnx_file)
