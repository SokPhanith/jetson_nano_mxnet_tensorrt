import math
import os
import sys
import time
from mxnet import autograd
from mxnet import gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo.vision import *
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import argparse
import numpy as np
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
def get_parser():
    parser = argparse.ArgumentParser(description="Fine tuning image classification.")
    parser.add_argument("--model",default="resnet18_v1",help="model for fine tuning.")
    parser.add_argument("--dataset_dir",default="data",help="path to data dir default[data].")
    parser.add_argument("--output_dir",default="output_training",help="path to output training default[output_training].")
    parser.add_argument("--cpu", action="store_true",help="set --cpu for training on cpu")
    parser.add_argument("--num_class",type=int,default=4,help="number classes default[4].")
    parser.add_argument("--epoch",type=int,default=5,help="number epoches training default[5].")
    parser.add_argument("--height",type=int,default=224,help="height for training default[244].")
    parser.add_argument("--width",type=int,default=224,help="width for training default[244].")
    parser.add_argument("--log_print",type=int,default=50,help="print loss while training default[50].")
    parser.add_argument("--lr",type=float,default=0.001,help="learning rate default[0.001].")
    parser.add_argument("--batch_per_device",type=int,default=8,help="batch size training default[4].")
    parser.add_argument("--num_workers",type=int,default=4,help="workers loading image default[4].")
    return parser

args = get_parser().parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if args.cpu:
    print('using cpu for training.')
    ctx = [mx.cpu()]
else:
    print('using gpu for training.')
    num_gpus = mx.context.num_gpus()
    print('number of gpu : ',num_gpus)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
batch_size = args.batch_per_device * max(num_gpus, 1)
print('Batch size for training : ',batch_size)

training_transformer = transforms.Compose([
    transforms.RandomResizedCrop(args.width),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
    transforms.RandomLighting(0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

validation_transformer = transforms.Compose([
    transforms.Resize(args.width),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

train_path = os.path.join(args.dataset_dir, 'train')
val_path = os.path.join(args.dataset_dir, 'valid')
test_path = os.path.join(args.dataset_dir, 'test')
list_label = sorted(os.listdir(train_path))
if not os.path.exists(args.output_dir+'/labels.txt'):
    with open(args.output_dir+'/labels.txt', 'w') as f:
        for single_label in list_label:
            f.write(single_label)
            f.write("\n")
    f.close()

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(training_transformer),
    batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(validation_transformer),
    batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

model_list = ['inception_v3','squeezenet1_1','resnet18_v1','resnet34_v1','resnet50_v1','resnet101_v1','resnet152_v1','resnet18_v2','resnet34_v2','resnet50_v2','resnet101_v2','resnet152_v2','alexnet','densenet121','densenet161','densenet169','densenet201','mobilenet0_25','mobilenet0_5','mobilenet0_75','mobilenet1_0','mobilenet_v2_0_25','mobilenet_v2_0_5','mobilenet_v2_0_75','mobilenet_v2_1_0','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','vgg11','vgg13','vgg16','vgg19']
pretrain_name = args.model
if pretrain_name == 'resnet18_v1':
    net = resnet18_v1(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet34_v1':
    net = resnet34_v1(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet50_v1':
    net = resnet50_v1(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet101_v1':
    net = resnet101_v1(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet152_v1':
    net = resnet152_v1(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet18_v2':
    net = resnet18_v2(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet34_v2':
    net = resnet34_v2(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet50_v2':
    net = resnet50_v2(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet101_v2':
    net = resnet101_v2(pretrained=True,ctx=ctx)
elif pretrain_name == 'resnet152_v2':
    net = resnet152_v2(pretrained=True,ctx=ctx)
elif pretrain_name == 'alexnet':
    net = alexnet(pretrained=True,ctx=ctx)
elif pretrain_name == 'densenet121':
    net = densenet121(pretrained=True,ctx=ctx)
elif pretrain_name == 'densenet161':
    net = densenet161(pretrained=True,ctx=ctx)
elif pretrain_name == 'densenet169':
    net = densenet169(pretrained=True,ctx=ctx)
elif pretrain_name == 'densenet201':
    net = densenet201(pretrained=True,ctx=ctx)
elif pretrain_name == 'inception_v3':
    net = inception_v3(pretrained=True,ctx=ctx)
    if args.width != 299 or args.height != 299:
        print("Fine tuning with inception_v3 must set width=height=299.")
        sys.exit()
elif pretrain_name == 'vgg11':
    net = vgg11(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg13':
    net = vgg13(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg16':
    net = vgg16(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg19':
    net = vgg19(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg11_bn':
    net = vgg11_bn(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg13_bn':
    net = vgg13_bn(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg16_bn':
    net = vgg16_bn(pretrained=True,ctx=ctx)
elif pretrain_name == 'vgg19_bn':
    net = vgg19_bn(pretrained=True,ctx=ctx)
elif pretrain_name == 'squeezenet1_0':
    net = squeezenet1_0(pretrained=True,ctx=ctx)
elif pretrain_name == 'squeezenet1_1':
    net = squeezenet1_1(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet0_25':
    net = mobilenet0_25(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet0_5':
    net = mobilenet0_5(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet0_75':
    net = mobilenet0_75(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet1_0':
    net = mobilenet1_0(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet_v2_0_25':
    net = mobilenet_v2_0_25(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet_v2_0_5':
    net = mobilenet_v2_0_5(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet_v2_0_75':
    net = mobilenet_v2_0_75(pretrained=True,ctx=ctx)
elif pretrain_name == 'mobilenet_v2_1_0':
    net = mobilenet_v2_1_0(pretrained=True,ctx=ctx)
else:
    for i in model_list:
        print(i)
    sys.exit()
    print('Unknow model from pretrain ImageNet mxnet.')
with net.name_scope():
    net.output = nn.Dense(args.num_class)
net.output.initialize(init.Xavier(), ctx=ctx)
net.hybridize()
print(net)
print('model fine tune : ',pretrain_name)

num_batch = len(train_data)
iterations_per_epoch = math.ceil(num_batch)
lr_steps = [epoch * iterations_per_epoch for epoch in [5,10,15]]
schedule = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=0.75, base_lr=args.lr)

sgd_optimizer = mx.optimizer.SGD(learning_rate=args.lr, lr_scheduler=schedule, momentum=0.9, wd=0.0001)
metric = mx.metric.Accuracy()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer=sgd_optimizer)

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for (data, label) in val_data:
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)
        outputs = [net(x) for x in data]
        metric.update(label, outputs)
    return metric.get()

for epoch in range(args.epoch):
    tic = time.time()
    train_loss = 0
    train_loss_log = 0
    metric.reset()
    for i, (data, label) in enumerate(train_data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, even_split=False)
        with autograd.record():
            outputs = [net(x) for x in data]
            loss = [softmax_cross_entropy(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()
        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss])/len(loss)
        metric.update(label, outputs)
        if i % args.log_print == 0:
            if i > 0:
                train_loss_log = train_loss
                current_loss = train_loss_log/i
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, i, current_loss))
            else:
                print("Epoch: %d; Batch %d; Loss %f" % (epoch, i, train_loss))
    _, train_acc = metric.get()
    train_loss /= num_batch
    _, val_acc = test(net, val_data, ctx)
    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | learning-rate: %.3E | time: %.1f' %(epoch, train_acc, train_loss, val_acc, trainer.learning_rate, time.time() - tic))
_, test_acc = test(net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
net.export(args.output_dir+"/"+pretrain_name+"_custom",epoch=args.epoch)
sym = args.output_dir+'/'+pretrain_name+'_custom-symbol.json'
onnx_file = args.output_dir+'/'+pretrain_name+'_custom.onnx'
if args.epoch <=9:
    params = args.output_dir+'/'+pretrain_name+'_custom-000'+str(args.epoch)+'.params'
elif 9 < args.epoch <= 99:
    params = args.output_dir+'/'+pretrain_name+'_custom-00'+str(args.epoch)+'.params'
else:
    params = args.output_dir+'/'+pretrain_name+'_custom-0'+str(args.epoch)+'.params'
print(sym)
print(params)
print(onnx_file)
if not os.path.exists(onnx_file):
    converted_model_path = onnx_mxnet.export_model(sym, params, [(1,3,args.width,args.height)], np.float32, onnx_file,)
