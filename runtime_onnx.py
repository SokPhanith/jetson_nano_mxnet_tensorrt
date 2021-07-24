import numpy as np
import cv2
import time
import onnxruntime
import argparse
from onnxruntime.datasets import get_example
import multiprocessing
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0):
    return ("nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=True"
            %(capture_width,
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
def get_parser():
    parser = argparse.ArgumentParser(description="TensorRT Runtime with mxnet Image_classification")
    parser.add_argument("--model",default="deploy.engine",help="path to tensorrt model.")
    parser.add_argument("--label",default="data/labels.txt",help="path to labels.txt one line label and no-background.")
    parser.add_argument("--csi", action="store_true",help="Set --csi for using pi-camera")
    parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
    parser.add_argument('--image', type=str, default=None,help='path to image file name')
    parser.add_argument("--video",type=str,default=None,help="Path to video file.")
    parser.add_argument("--height",type=int,default=224,help="height input image default[224].")
    parser.add_argument("--width",type=int,default=224,help="width input image default[224].")
    parser.add_argument("--height_display",type=int,default=480,help="height display image default[480].")
    parser.add_argument("--width_display",type=int,default=640,help="width display image default[640].")
    parser.add_argument("--batch_size",type=int,default=1,help="batch size input image default[1].")
    parser.add_argument("--channel",type=int,default=3,help="channel input image default[3].")
    return parser
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
args = get_parser().parse_args()
mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32) 
std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)  
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
print('model :',args.model)
session = onnxruntime.InferenceSession(args.model)
print("The model expects input shape: ", session.get_inputs()[0].shape)
input_name = session.get_inputs()[0].name
print('label :',args.label)
with open(args.label) as labels:
    labels = [i.strip() for i in labels.readlines()]
while True:
    if args.image:
        frame = cv2.imread(args.image)
    else:
        c , frame = cam.read()
        if c == False:
            break
    resized = cv2.resize(frame, (224, 224))
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = (img_in.astype('float32')/255.0 - mean)/std
    img_in = img_in.transpose((2, 0, 1))
    img_in = np.expand_dims(img_in, axis=0)
    t1 = time.time()
    [[outputs]] = session.run(None, {input_name: img_in})
    dt = time.time() - t1
    output_softmax = softmax(outputs)
    pred_index = output_softmax.argmax()
    classes = labels[pred_index]
    percent = output_softmax[pred_index]
    cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (11, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32, 32), 4, cv2.LINE_AA)
    cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(round(1.0/dt,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32),4, cv2.LINE_AA)
    cv2.putText(frame, str(round(1.0/dt,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)   
    cv2.imshow('Demo',frame)
    cv2.moveWindow('Demo',0,0)
    if cv2.waitKey(1) == ord('q'):
        break
if args.image:
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()
    cam.release()

