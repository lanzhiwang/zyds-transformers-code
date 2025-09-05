FROM jupyter/minimal-notebook:x86_64-python-3.11.6

RUN pip -v install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip -v install \
        transformers==4.42.4 \
        peft==0.11.1 \
        datasets==2.20.0 \
        accelerate==0.32.1 \
        bitsandbytes==0.43.1 \
        faiss-cpu==1.7.4 \
        tensorboard==2.14.0 \
        modelscope \
        "black[jupyter]" && \
    pip cache purge
