import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import sys
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Build TensorRT engine model")
    parser.add_argument("--model",default="model.onnx",help="path to onnx model.")
    parser.add_argument("--output_folder",default="tensorrt_model",help="folder put tensorrt engine.")
    parser.add_argument("--fp16", action="store_true",help="Set --fp16 for build tensorrt FP16 default[FP32]")
    parser.add_argument("--batch_size",type=int,default=1,help="batch size input image default[1].")
    return parser
def set_net_batch(network, batch_size):
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network
def build_engine_onnx(model_file,type_engine,batch_size,path_model):
    if type_engine:
        onnx_file = str(model_file.split("/")[-1])
        model_name = str(onnx_file.split(".")[0])+'_fp16.engine'
    else:
        onnx_file = str(model_file.split("/")[-1])
        model_name = str(onnx_file.split(".")[0])+'_fp32.engine'
    if os.path.exists(path_model+"/"+model_name):
        print('TensorRT Engine model was build.')
        sys.exit()
    with open(model_file, 'rb') as f:
        model = f.read()
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = [] if trt.__version__[0] < '7' else [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not parser.parse(model):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network,batch_size)
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        if type_engine:
            config.set_flag(trt.BuilderFlag.FP16)
        engine = builder.build_engine(network, config)
        with open(path_model+"/"+model_name, 'wb') as f:
            f.write(engine.serialize())
args = get_parser().parse_args()    
folder_tensorrt_model = args.output_folder
if not os.path.exists(folder_tensorrt_model):
    os.makedirs(folder_tensorrt_model)
build_engine_onnx(args.model,args.fp16,args.batch_size,folder_tensorrt_model)