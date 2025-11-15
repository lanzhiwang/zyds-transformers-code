"""
文本分类实例
"""

import time
import math

import pandas as pd

import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from accelerate import Accelerator

# 模型保存
from peft import LoraConfig, get_peft_model

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

    # 模型保存
    lora_config = LoraConfig(target_modules=["query", "key", "value"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


# def train(
#     model,
#     optimizer,
#     trainloader,
#     validloader,
#     accelerator: Accelerator,
#     epoch=3,
#     log_step=10,
# ):
def train(
    model,
    optimizer,
    trainloader,
    validloader,
    accelerator: Accelerator,
    resume,
    epoch=3,
    log_step=10,
):
    global_step = 0
    start_time = time.time()

    resume_step = 0
    resume_epoch = 0

    # 断点续训
    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = math.ceil(
            len(trainloader) / accelerator.gradient_accumulation_steps
        )
        resume_step = global_step = int(resume.split("step_")[-1])
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from checkpoint -> {resume}")

    # for ep in range(epoch):
    for ep in range(resume_epoch, epoch):
        model.train()

        # 断点续训
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(
                trainloader, resume_step * accelerator.gradient_accumulation_steps
            )
        else:
            active_dataloader = trainloader

        # for batch in trainloader:
        for batch in active_dataloader:
            # 梯度累积
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()

                # 梯度累积
                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(
                            f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}"
                        )
                        # 实验记录
                        accelerator.log({"loss": loss.item()}, global_step)

                    # 模型保存和断点续训
                    if global_step % 50 == 0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        # 断点续训
                        accelerator.save_state(
                            accelerator.project_dir + f"/step_{global_step}"
                        )
                        # 模型保存
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir
                            + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save,
                        )
                        # accelerator.save_model(model, accelerator.project_dir + f"/step_{global_step}")

        acc = evaluate(model, validloader, accelerator)
        # accelerator.print(f"ep: {ep}, acc: {acc}")
        accelerator.print(f"ep: {ep}, acc: {acc}, time: {time.time() - start_time}")
        # 实验记录
        accelerator.log({"acc": acc}, global_step)

    # 实验记录
    accelerator.end_training()


def main():

    # accelerator = Accelerator()
    # 混合精度 mixed_precision
    # 梯度累积 gradient_accumulation_steps
    # 实验记录 log_with, project_dir
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=2,
        log_with="tensorboard",
        project_dir="ckpts",
    )

    # 实验记录
    accelerator.init_trackers("runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(
        model, optimizer, trainloader, validloader
    )

    # 断点续训
    # train(model, optimizer, trainloader, validloader, accelerator)
    train(model, optimizer, trainloader, validloader, accelerator, resume=None)
    # train(
    #     model,
    #     optimizer,
    #     trainloader,
    #     validloader,
    #     accelerator,
    #     resume="/root/LLaMA-Factory/32_accelerate_advanced/ckpts/step_150",
    # )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES="2,3" torchrun --nproc_per_node=2 01_ddp_accelerate_advanced.py
# CUDA_VISIBLE_DEVICES="2,3" accelerate launch 01_ddp_accelerate_advanced.py
# CUDA_VISIBLE_DEVICES="2,3" accelerate launch --mixed_precision bf16 01_ddp_accelerate_advanced.py

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
bf16
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
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
$

$ tree -a ckpts/
ckpts/
└── runs
    └── events.out.tfevents.1763177977.nj-p-zdtc-h20-059.966775.0

1 directory, 1 file
$

$ tree -a ckpts/
ckpts/
├── runs
│   ├── events.out.tfevents.1763177977.nj-p-zdtc-h20-059.966775.0
│   └── events.out.tfevents.1763178769.nj-p-zdtc-h20-059.976565.0
├── step_100
│   └── model.safetensors
├── step_150
│   └── model.safetensors
└── step_50
    └── model.safetensors

4 directories, 5 files
$

$ tree -a ckpts/
ckpts/
├── runs
│   └── events.out.tfevents.1763179589.nj-p-zdtc-h20-059.988673.0
├── step_100
│   └── model
│       ├── config.json
│       └── model.safetensors
├── step_150
│   └── model
│       ├── config.json
│       └── model.safetensors
└── step_50
    └── model
        ├── config.json
        └── model.safetensors

7 directories, 7 files
$

$ tree -a ckpts/
ckpts/
├── runs
│   └── events.out.tfevents.1763179445.nj-p-zdtc-h20-059.986045.0
├── step_100
│   └── model
│       ├── README.md
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── step_150
│   └── model
│       ├── README.md
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── step_50
    └── model
        ├── README.md
        ├── adapter_config.json
        └── adapter_model.safetensors

7 directories, 10 files
$

$ tree -a ckpts/
ckpts/
├── runs
│   └── events.out.tfevents.1763179914.nj-p-zdtc-h20-059.993257.0
├── step_100
│   ├── model
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── model.safetensors
│   ├── optimizer.bin
│   ├── random_states_0.pkl
│   └── random_states_1.pkl
├── step_150
│   ├── model
│   │   ├── README.md
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── model.safetensors
│   ├── optimizer.bin
│   ├── random_states_0.pkl
│   └── random_states_1.pkl
└── step_50
    ├── model
    │   ├── README.md
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── model.safetensors
    ├── optimizer.bin
    ├── random_states_0.pkl
    └── random_states_1.pkl

7 directories, 22 files
$

"""
