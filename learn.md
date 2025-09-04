```bash
############ 启动基本服务
docker pull jupyter/minimal-notebook:x86_64-python-3.11.6

docker run -it --rm --name "huggingface" \
-p 10000:8888 \
--gpus all \
-v ~/zyds-transformers-code:/home/jovyan/work/zyds-transformers-code \
jupyter/minimal-notebook:x86_64-python-3.11.6

docker run -it --rm --name "huggingface" \
-p 10000:8888 \
--gpus all \
-v ~/zyds-transformers-code:/home/jovyan/work/zyds-transformers-code \
lanzhiwang/minimal-notebook-x86_64-python-3-11-6:sha-18ca95a

http://192.168.100.59:10000/lab?token=a6795bc13ab756d8110f8ce6c985f3f537e8d9a394d0b7ab

############ 验证 GPU 是否可用
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip -v install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip -v install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

$ python
Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
8
>>>

############ 安装相应依赖

uv pip -v install transformers==4.42.4 peft==0.11.1 datasets==2.20.0 accelerate==0.32.1 bitsandbytes==0.43.1 faiss-cpu==1.7.4 tensorboard==2.14.0

uv pip -v install "black[jupyter]"
uv pip -v install modelscope "black[jupyter]"
```
