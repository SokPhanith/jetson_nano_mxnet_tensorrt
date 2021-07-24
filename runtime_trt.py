import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import sys 
import cv2 
import time
import numpy as np
import argparse
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
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

if __name__ == "__main__":
    args = get_parser().parse_args()
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32) 
    std =np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)  
    fps_old = 0
    count = 0
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
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    window_name = args.model.split('/')[-1].split('.')[0]
    if window_name == 'inception_v3_fp16' or window_name == 'inception_v3_fp32':
        if args.height != 299 or args.width != 299:
            print('inception_v3 must input size 299x299x3')
            sys.exit()
    print('model :',args.model)
    with open(args.model, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
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
        img = cv2.resize(frame,(args.width,args.height))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = (img.astype('float32')/255.0 - mean)/std
        img = img.transpose((2, 0, 1))
        data = np.reshape(img,(args.channel*args.height*args.width))  
        inputs[0].host = data
        t1 = time.time()
        [trt_outputs] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,batch_size=args.batch_size)
        dt = time.time() - t1
        softmax_output = softmax(trt_outputs)
        pred_index = softmax_output.argmax()
        classes = labels[pred_index]
        percent = softmax_output[pred_index]
        if count < 10:
            Fps = (1.0/dt)
            count += 1
        else:       
            Fps = 0.95*fps_old + 0.05*(1.0/dt)
        fps_old = Fps
        cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (11, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32, 32), 4, cv2.LINE_AA)
        cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(round(Fps,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32),4, cv2.LINE_AA)
        cv2.putText(frame, str(round(Fps,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (240, 202, 0), 1, cv2.LINE_AA)   
        cv2.imshow(window_name,frame)
        cv2.moveWindow(window_name,0,0)
        if cv2.waitKey(1) == ord('q'):
            break
    if args.image:
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        cam.release()
