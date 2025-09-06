```bash
############ 启动基本服务
docker pull jupyter/minimal-notebook:x86_64-python-3.11.6

docker run -it --rm --name "huggingface" \
-p 10000:8888 \
--gpus all \
-v ~/zyds-transformers-code:/home/jovyan/work/zyds-transformers-code \
jupyter/minimal-notebook:x86_64-python-3.11.6

docker run -it --rm \
-u root \
--name "huggingface" \
-p 10000:8888 \
--gpus all \
-v /root/zyds-transformers-code:/home/jovyan/work/zyds-transformers-code \
lanzhiwang/minimal-notebook-x86_64-python-3-11-6:sha-1493536

http://192.168.100.59:10000/lab?token=9dba172f6929c5bf6fb3ee61e256d6b40095b848d5a31b88

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

uv pip -v install transformers==4.42.4 peft==0.11.1 datasets==2.20.0 accelerate==0.32.1 bitsandbytes==0.43.1 faiss-cpu==1.7.4 tensorboard==2.14.0 evaluate

uv pip -v install modelscope "black[jupyter]"

############ 下载模型和数据集

export HF_ENDPOINT=https://hf-mirror.com

# 下载模型
HF_ENDPOINT=https://hf-mirror.com hf download Langboat/bloom-1b4-zh --local-dir ./models/Langboat/bloom-1b4-zh/

modelscope download --model Langboat/bloom-1b4-zh --local_dir ./models/Langboat/bloom-1b4-zh/

# 下载数据集
HF_ENDPOINT=https://hf-mirror.com hf download --repo-type dataset wikitext --local-dir ./dataset/wikitext

```
