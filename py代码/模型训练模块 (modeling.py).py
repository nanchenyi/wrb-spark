import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from accelerate import Accelerator
import time
import numpy as np


class GLUEDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]
        self.token_type_ids = dataset["token_type_ids"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "token_type_ids": torch.tensor(self.token_type_ids[idx]),
            "labels": torch.tensor(self.labels[idx])
        }


class GLUEClassifierTrainer:
    def __init__(self, model_name, task_name, max_length=128,
                 batch_size=32, learning_rate=2e-5, epochs=3):
        self.model_name = model_name
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_time = 0
        self.max_gpu_memory = 0

    def train(self, train_data, val_data):
        start_time = time.time()

        # 1. 准备数据集
        train_dataset = GLUEDataset(train_data)
        val_dataset = GLUEDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2)

        # 2. 初始化模型
        num_labels = 1 if self.task_name == "stsb" else 2
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

        # 3. 准备优化器
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        # 4. 使用Accelerate处理设备分配
        accelerator = Accelerator()
        device = accelerator.device

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        # 5. 训练循环
        print(f"开始训练: {self.model_name} | 批次: {self.batch_size} | 学习率: {self.learning_rate}")

        for epoch in range(self.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()

                # 监控GPU显存
                if torch.cuda.is_available():
                    self.max_gpu_memory = max(
                        self.max_gpu_memory,
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    )

            # 验证集评估
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == batch["labels"]).sum().item()
                    total += batch["labels"].size(0)

            accuracy = correct / total
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"训练损失: {total_loss / len(train_loader):.4f} | "
                  f"验证损失: {val_loss / len(val_loader):.4f} | "
                  f"验证准确率: {accuracy:.4f} | "
                  f"耗时: {epoch_time:.2f}s")

        self.training_time = time.time() - start_time
        print(f"训练完成! 总耗时: {self.training_time:.2f}秒")

        # 保存训练好的模型
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        return unwrapped_model, AutoTokenizer.from_pretrained(self.model_name)