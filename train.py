from peft import LoraConfig, TaskType 
import datasets
from datasets import load_dataset, load_from_disk
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer

dataset = load_from_disk("textdata")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)
from datasets import concatenate_datasets
import numpy as np

# Max input sequence length post-tokenization.
max_input_length = 512
max_target_length = 50

def preprocess_function(examples):
    inputs = [ex for ex in examples['dialogue']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setting up the tokenizer for targets
    labels = tokenizer(examples['summary'], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_datasets = dataset.map(preprocess_function, batched=True)


for example in dataset[ "train"]:
    # 提取對話和摘要
    dialogue = example['dialogue']
    summary = example['summary']
    print(f"dialogue = \n{dialogue}")
    print(f"summary = \n{summary}")
    break

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()