"""
# 文本分类实例
"""

## Step1 导入相关包
import os
import torch

# 使用 transformers 相关的库时, 环境变量 CUDA_VISIBLE_DEVICES 要在导入 transformers 库之前设置
print("device_count1:", torch.cuda.device_count())  # device_count1: 8
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print("device_count2:", torch.cuda.device_count())  # device_count2: 2

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

## Step2 加载数据集
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)

## Step3 划分数据集
datasets = dataset.train_test_split(test_size=0.1)

## Step4 数据集预处理
tokenizer = AutoTokenizer.from_pretrained("/root/LLaMA-Factory/models/hfl/rbt3")


def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples


tokenized_datasets = datasets.map(
    process_function, batched=True, remove_columns=datasets["train"].column_names
)

## Step5 创建模型
model = AutoModelForSequenceClassification.from_pretrained(
    "/root/LLaMA-Factory/models/hfl/rbt3", device_map="auto"
)
print("model:", model)
print("model.device:", model.device)

## Step6 创建 TrainingArguments
train_args = TrainingArguments(
    # 输出文件夹
    output_dir="./checkpoints",
    # 训练时的 batch_size
    per_device_train_batch_size=64,
    # 验证时的 batch_size
    per_device_eval_batch_size=128,
    # log 打印的频率
    logging_steps=10,
    # 评估策略
    eval_strategy="epoch",
    # 保存策略
    save_strategy="epoch",
    # 最大保存数
    save_total_limit=3,
    # 学习率
    learning_rate=2e-5,
    # weight_decay
    weight_decay=0.01,
    # 设定评估指标
    metric_for_best_model="eval_loss",
    # 训练完成后加载最优模型
    load_best_model_at_end=True,
    report_to="none",
)
print("train_args._n_gpu:", train_args._n_gpu)

## Step7 创建 Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    # compute_metrics=eval_metric,
)

## Step8 模型训练
trainer.train()

## Step9 模型评估
trainer.evaluate(tokenized_datasets["test"])
