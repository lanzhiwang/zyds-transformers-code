```bash
docker pull jupyter/minimal-notebook:x86_64-python-3.11.6

docker run -it --rm -p 10000:8888 \
-v ~/work/code/go_code/ai/huggingface/zyds-transformers-code:/home/jovyan/work/zyds-transformers-code \
jupyter/minimal-notebook:x86_64-python-3.11.6


- torch==2.2.1+cu118

- transformers==4.42.4

- peft==0.11.1

- datasets==2.20.0

- accelerate==0.32.1

- bitsandbytes==0.43.1

- faiss-cpu==1.7.4

- tensorboard==2.14.0



```
