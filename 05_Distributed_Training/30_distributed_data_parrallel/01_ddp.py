"""
文本分类实例
"""

import os

print("os.environ:", os.environ)
# os.environ: environ({
#     'CUDA_VISIBLE_DEVICES': '2,3',
#     'SHELL': '/bin/bash',
#     'ROS_VERSION': '2',
#     'CONDA_EXE': '/root/miniconda3/bin/conda',
#     'ROS_PYTHON_VERSION': '3',
#     'OLLAMA_MODELS': '/home/zhanghao/edu/ollama/model',
#     'PWD': '/root/LLaMA-Factory/30_distributed_data_parrallel',
#     'LOGNAME': 'root',
#     'XDG_SESSION_TYPE': 'tty',
#     'CONDA_PREFIX': '/root/miniconda3/envs/llamafactory',
#     'MOTD_SHOWN': 'pam',
#     'HOME': '/root',
#     'LANG': 'C.UTF-8',
#     'CONDA_PROMPT_MODIFIER': '(llamafactory) ',
#     'AMENT_PREFIX_PATH': '/opt/ros/humble',
#     'LC_TERMINAL': 'iTerm2',
#     'SSH_CONNECTION': '172.16.68.122 53182 192.168.100.59 22',
#     'LESSCLOSE': '/usr/bin/lesspipe %s %s',
#     'XDG_SESSION_CLASS': 'user',
#     'PYTHONPATH': '/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages',
#     'TERM': 'xterm-256color',
#     'LESSOPEN': '| /usr/bin/lesspipe %s',
#     'USER': 'root',
#     'CONDA_SHLVL': '2',
#     'LC_TERMINAL_VERSION': '3.3.7',
#     'SHLVL': '1',
#     'XDG_SESSION_ID': '2816',
#     'CONDA_PYTHON_EXE': '/root/miniconda3/bin/python',
#     'LD_LIBRARY_PATH': '/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib',
#     'XDG_RUNTIME_DIR': '/run/user/0',
#     'ROS_LOCALHOST_ONLY': '0',
#     'SSH_CLIENT': '172.16.68.122 53182 22',
#     'CONDA_DEFAULT_ENV': 'llamafactory',
#     'LC_TIME': 'C.UTF-8',
#     'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop',
#     'PATH': '/opt/ros/humble/bin:/root/miniconda3/envs/llamafactory/bin:/root/miniconda3/condabin:/sbin:/usr/sbin:/usr/local/sbin:/usr/lpp/mmfs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
#     'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/0/bus',
#     'SSH_TTY': '/dev/pts/1',
#     'CONDA_PREFIX_1': '/root/miniconda3',
#     'ROS_DISTRO': 'humble',
#     'OLDPWD': '/root/LLaMA-Factory',
#     '_': '/root/miniconda3/envs/llamafactory/bin/torchrun',
#     'OMP_NUM_THREADS': '1',
#     'LOCAL_RANK': '0',  ######
#     'RANK': '0',  ######
#     'GROUP_RANK': '0',
#     'ROLE_RANK': '0',  ######
#     'ROLE_NAME': 'default',
#     'LOCAL_WORLD_SIZE': '2',
#     'WORLD_SIZE': '2',
#     'GROUP_WORLD_SIZE': '1',
#     'ROLE_WORLD_SIZE': '2',
#     'MASTER_ADDR': '127.0.0.1',
#     'MASTER_PORT': '29500',
#     'TORCHELASTIC_RESTART_COUNT': '0',
#     'TORCHELASTIC_MAX_RESTARTS': '0',
#     'TORCHELASTIC_RUN_ID': 'none',
#     'TORCHELASTIC_USE_AGENT_STORE': 'True',
#     'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
#     'TORCHELASTIC_ERROR_FILE': '/tmp/torchelastic_vas9_dod/none_77dl2omi/attempt_0/0/error.json'  ######
# })

# os.environ: environ({
#     'CUDA_VISIBLE_DEVICES': '2,3',
#     'SHELL': '/bin/bash',
#     'ROS_VERSION': '2',
#     'CONDA_EXE': '/root/miniconda3/bin/conda',
#     'ROS_PYTHON_VERSION': '3',
#     'OLLAMA_MODELS': '/home/zhanghao/edu/ollama/model',
#     'PWD': '/root/LLaMA-Factory/30_distributed_data_parrallel',
#     'LOGNAME': 'root',
#     'XDG_SESSION_TYPE': 'tty',
#     'CONDA_PREFIX': '/root/miniconda3/envs/llamafactory',
#     'MOTD_SHOWN': 'pam',
#     'HOME': '/root',
#     'LANG': 'C.UTF-8',
#     'CONDA_PROMPT_MODIFIER': '(llamafactory) ',
#     'AMENT_PREFIX_PATH': '/opt/ros/humble',
#     'LC_TERMINAL': 'iTerm2',
#     'SSH_CONNECTION': '172.16.68.122 53182 192.168.100.59 22',
#     'LESSCLOSE': '/usr/bin/lesspipe %s %s',
#     'XDG_SESSION_CLASS': 'user',
#     'PYTHONPATH': '/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages',
#     'TERM': 'xterm-256color',
#     'LESSOPEN': '| /usr/bin/lesspipe %s',
#     'USER': 'root',
#     'CONDA_SHLVL': '2',
#     'LC_TERMINAL_VERSION': '3.3.7',
#     'SHLVL': '1',
#     'XDG_SESSION_ID': '2816',
#     'CONDA_PYTHON_EXE': '/root/miniconda3/bin/python',
#     'LD_LIBRARY_PATH': '/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib',
#     'XDG_RUNTIME_DIR': '/run/user/0',
#     'ROS_LOCALHOST_ONLY': '0',
#     'SSH_CLIENT': '172.16.68.122 53182 22',
#     'CONDA_DEFAULT_ENV': 'llamafactory',
#     'LC_TIME': 'C.UTF-8',
#     'XDG_DATA_DIRS': '/usr/local/share:/usr/share:/var/lib/snapd/desktop',
#     'PATH': '/opt/ros/humble/bin:/root/miniconda3/envs/llamafactory/bin:/root/miniconda3/condabin:/sbin:/usr/sbin:/usr/local/sbin:/usr/lpp/mmfs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
#     'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/0/bus',
#     'SSH_TTY': '/dev/pts/1',
#     'CONDA_PREFIX_1': '/root/miniconda3',
#     'ROS_DISTRO': 'humble',
#     'OLDPWD': '/root/LLaMA-Factory',
#     '_': '/root/miniconda3/envs/llamafactory/bin/torchrun',
#     'OMP_NUM_THREADS': '1',
#     'LOCAL_RANK': '1',  ######
#     'RANK': '1',  ######
#     'GROUP_RANK': '0',
#     'ROLE_RANK': '1',  ######
#     'ROLE_NAME': 'default',
#     'LOCAL_WORLD_SIZE': '2',
#     'WORLD_SIZE': '2',
#     'GROUP_WORLD_SIZE': '1',
#     'ROLE_WORLD_SIZE': '2',
#     'MASTER_ADDR': '127.0.0.1',
#     'MASTER_PORT': '29500',
#     'TORCHELASTIC_RESTART_COUNT': '0',
#     'TORCHELASTIC_MAX_RESTARTS': '0',
#     'TORCHELASTIC_RUN_ID': 'none',
#     'TORCHELASTIC_USE_AGENT_STORE': 'True',
#     'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
#     'TORCHELASTIC_ERROR_FILE': '/tmp/torchelastic_vas9_dod/none_77dl2omi/attempt_0/1/error.json'  ######
# })

import pandas as pd

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


## Step1 初始化进程组
dist.init_process_group(backend="nccl")


## Step2 加载数据
data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
data = data.dropna()  # 去掉空行


## Step3 创建 Dataset
class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


dataset = MyDataset()

## Step4 划分数据集
trainset, validset = random_split(dataset, lengths=[0.9, 0.1])


## Step5 创建 Dataloader
tokenizer = AutoTokenizer.from_pretrained("/root/LLaMA-Factory/models/hfl/rbt3")


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(labels)
    return inputs


trainloader = DataLoader(
    trainset,
    batch_size=32,
    collate_fn=collate_func,
    sampler=DistributedSampler(trainset),
)

validloader = DataLoader(
    validset,
    batch_size=64,
    collate_fn=collate_func,
    sampler=DistributedSampler(validset),
)


## Step6 创建模型及优化器
model = AutoModelForSequenceClassification.from_pretrained(
    "/root/LLaMA-Factory/models/hfl/rbt3"
)
if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))

model = DistributedDataParallel(model)

optimizer = Adam(model.parameters(), lr=2e-5)


## Step7 训练与验证
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {
                    k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
                }
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            print(f'evaluate pred: {pred.long()}, labels: {batch["labels"].long()}')
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)


def train(epoch=3, log_step=10):
    global_step = 0

    for ep in range(epoch):
        model.train()
        # 每个 epoch 训练全部数据
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {
                    k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
                }
            optimizer.zero_grad()
            output = model(**batch)
            print("output.loss: ", output.loss)
            output.loss.backward()
            optimizer.step()

            if global_step % log_step == 0:
                print(
                    f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}"
                )

            global_step += 1

        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")


train()

# CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node=2 01_ddp.py
