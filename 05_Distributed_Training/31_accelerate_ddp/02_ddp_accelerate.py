"""
文本分类实例
"""

# import os

import pandas as pd

import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel

from accelerate import Accelerator

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


def prepare_dataloader():

    dataset = MyDataset()

    trainset, validset = random_split(
        dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )

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

    # trainloader = DataLoader(
    #     trainset,
    #     batch_size=32,
    #     collate_fn=collate_func,
    #     sampler=DistributedSampler(trainset),
    # )
    # validloader = DataLoader(
    #     validset,
    #     batch_size=64,
    #     collate_fn=collate_func,
    #     sampler=DistributedSampler(validset),
    # )
    trainloader = DataLoader(
        trainset, batch_size=32, collate_fn=collate_func, shuffle=True
    )
    validloader = DataLoader(
        validset, batch_size=64, collate_fn=collate_func, shuffle=False
    )

    return trainloader, validloader


def prepare_model_and_optimizer():

    model = AutoModelForSequenceClassification.from_pretrained(
        "/root/LLaMA-Factory/models/hfl/rbt3"
    )

    # if torch.cuda.is_available():
    #     model = model.to(int(os.environ["LOCAL_RANK"]))

    # model = DistributedDataParallel(model)

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


# def print_rank_0(info):
#     if int(os.environ["RANK"]) == 0:
#         print(info)


# def evaluate(model, validloader):
def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            # if torch.cuda.is_available():
            #     batch = {
            #         k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
            #     }
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # print(f"evaluate: pred: {pred}")
            # print(f"evaluate: pred: {pred.shape}")
            # acc_num += (pred.long() == batch["labels"].long()).float().sum()
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    # dist.all_reduce(acc_num)
    # print(f"evaluate: pred: {len(validloader.dataset)}")
    return acc_num / len(validloader.dataset)


# def train(model, optimizer, trainloader, validloader, epoch=3, log_step=100):
def train(
    model,
    optimizer,
    trainloader,
    validloader,
    accelerator: Accelerator,
    epoch=3,
    log_step=10,
):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            # if torch.cuda.is_available():
            #     batch = {
            #         k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()
            #     }
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            if global_step % log_step == 0:
                # dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                loss = accelerator.reduce(loss, "mean")
                # print_rank_0(
                #     f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                # )
                accelerator.print(
                    f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                )
            global_step += 1
        acc = evaluate(model, validloader, accelerator)
        # print_rank_0(f"ep: {ep}, acc: {acc}")
        accelerator.print(f"ep: {ep}, acc: {acc}")


def main():

    # dist.init_process_group(backend="nccl")
    accelerator = Accelerator()

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(
        model, optimizer, trainloader, validloader
    )

    # train(model, optimizer, trainloader, validloader)
    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node=2 02_ddp_accelerate.py
# CUDA_VISIBLE_DEVICES="2,3" accelerate launch 02_ddp_accelerate.py

"""
$ accelerate config
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:2,3
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
no
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
$

$ cat /root/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 2,3
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
$

$ accelerate launch --help

$ CUDA_VISIBLE_DEVICES="2,3" accelerate launch --config_file accelerate/default_config.yaml 02_ddp_accelerate.py

"""
