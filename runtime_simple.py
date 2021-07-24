from mxnet import nd
import mxnet as mx
import cv2
from collections import namedtuple
import time
import os
import sys
import argparse
import numpy as np
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
def get_parser():
    parser = argparse.ArgumentParser(description="Runtime simple with mxnet Image_classification")
    parser.add_argument("--model",default="deploy.engine",help="path to tensorrt model.")
    parser.add_argument("--label",default="data/labels.txt",help="path to labels.txt one line label and no-background.")
    parser.add_argument("--csi", action="store_true",help="Set --csi for using pi-camera")
    parser.add_argument("--cpu", action="store_true",help="Set --cpu for cpu inference default[gpu]")
    parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
    parser.add_argument('--image', type=str, default=None,help='path to image file name')
    parser.add_argument("--video",type=str,default=None,help="Path to video file.")
    parser.add_argument("--height",type=int,default=224,help="height input image default[224].")
    parser.add_argument("--epoch",type=int,default=0,help="epoch model export default[0].")
    parser.add_argument("--width",type=int,default=224,help="width input image default[224].")
    parser.add_argument("--height_display",type=int,default=480,help="height display image default[480].")
    parser.add_argument("--width_display",type=int,default=640,help="width display image default[640].")
    parser.add_argument("--batch_size",type=int,default=1,help="batch size input image default[1].")
    parser.add_argument("--channel",type=int,default=3,help="channel input image default[3].")
    return parser
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    port=0,
    flip_method=0):
    return ("nvarguscamerasrc sensor_id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=True"
            %(  port,
		capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,))
def open_cam_usb(dev,width,height,USB_GSTREAMER):
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture('/dev/video'+str(dev))
def transform(data,ctx):
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406],ctx=ctx).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225],ctx=ctx).reshape((1,3,1,1))
    return (data.astype('float32')/255.0 - rgb_mean) / rgb_std
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def preprocess(img,width=224,height=224,do_cropping=True):
    crop = img
    if do_cropping:
        h,w,_ = img.shape
        if h < w:
            crop = img[:, ((w-h)//2):((w+h)//2), :]
        else:
            crop = img[((h-w)//2):((h+w)//2), :, :]
    crop = cv2.resize(crop, (width,height))
    crop = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    return crop

if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.csi:
        print("csi using")
        cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0,display_width=args.width_display,display_height=args.height_display),cv2.CAP_GSTREAMER)
    elif args.image:
        print("image for classification")
        print(args.image)
    elif args.webcam:
        print('webcam using')
        cam = open_cam_usb(int(args.webcam),args.width_display,args.height_display,USB_GSTREAMER=True)
    elif args.video:
        print('video for classification')
        cam = cv2.VideoCapture(args.video)
    else:
        print('None source for input need image, video, csi or webcam')
        sys.exit()
    if args.cpu:
        print('using cpu for inference')
        ctx = mx.cpu()
    else:
        print('using gpu for inference')
        ctx = mx.gpu()
    if len(args.model.split("-")) == 2:
        split_model_name = args.model.split("-")[1]
        if split_model_name == '0000.params' or split_model_name == 'symbol.json':
            name_model_load = args.model.split("-")[0]
    else:
        name_model_load = args.model
    window_name = name_model_load.split('/')[-1].split('.')[0]
    if window_name == 'inception_v3':
        if args.height != 299 or args.width != 299:
            print('inception_v3 must input size 299x299x3')
            sys.exit()
    print('model : ',name_model_load)
    sym, arg_params, aux_params = mx.model.load_checkpoint(name_model_load,epoch=args.epoch)
    net = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    net.bind(for_training=False, data_shapes=[('data', (args.batch_size,args.channel, args.width,args.height))], label_shapes=net._label_shapes)
    net.set_params(arg_params, aux_params, allow_missing=True)
    print('label :',args.label)
    with open(args.label) as labels:
        labels = [i.strip() for i in labels.readlines()]
    Batch = namedtuple('Batch', ['data'])
    while True:
        if args.image:
            frame = cv2.imread(args.image)
        else:
            c,frame = cam.read()
            if c == False:
                break
        x = preprocess(frame,width=args.width,height=args.height,do_cropping=False)
        x = transform(nd.array(x,ctx=ctx),ctx=ctx)
        t1 = time.time()
        net.forward(Batch([x]))
        prob = net.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        dt = time.time() - t1
        prob = softmax(prob)
        idx = np.argmax(prob)
        percent = prob[idx]
        label = labels[idx]
        cv2.putText(frame,'{} {:.2f}'.format(label, round(percent*100,2)), (11, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32, 32), 4, cv2.LINE_AA)
        cv2.putText(frame,'{} {:.2f}'.format(label, round(percent*100,2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(round(1.0/dt,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32),4, cv2.LINE_AA)
        cv2.putText(frame, str(round(1.0/dt,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)   
        cv2.imshow(window_name,frame)
        cv2.moveWindow(window_name,0,0)
        if cv2.waitKey(1) == ord('q'):
            break
    if args.image:
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        cam.release()


    
