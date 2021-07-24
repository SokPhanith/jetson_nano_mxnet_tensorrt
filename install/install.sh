# 1. Install dependencies
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y git build-essential libopenblas-base python3-pip libprotobuf-dev protobuf-compiler libopencv-dev graphviz libopenblas-dev libopenblas-base libatlas-base-dev libprotoc-dev python-setuptools

# 2. Install pip prerequisite
sudo pip3 install setuptools
sudo pip3 install numpy==1.19.4 graphviz protobuf cython
sudo pip3 install onnx==1.3.0

# 3. Install mxnet pre-build
sudo pip3 install https://mxnet-public.s3.amazonaws.com/install/jetson/1.6.0/mxnet_cu102_arch53-1.6.0-py2.py3-none-linux_aarch64.whl

# 4. install onnxruntime_gpu
sudo pip3 install onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl
