from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import numpy as np


class ModelEvaluator:
    def __init__(self, model, tokenizer, task_name, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self, test_data):
        # 准备测试数据集
        test_dataset = GLUEDataset(test_data)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # 预测循环
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # 打印结果
        print("\n" + "=" * 50)
        print(f"{self.task_name}测试结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("混淆矩阵:")
        print(conf_matrix)

        # 错误分析示例
        self.error_analysis(test_data, all_labels, all_preds)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }

    def error_analysis(self, test_data, labels, preds):
        print("\n错误分析示例:")
        incorrect_indices = [i for i, (l, p) in enumerate(zip(labels, preds)) if l != p]

        # 打印前5个错误样本
        for idx in incorrect_indices[:5]:
            if self.task_name == "sst2":
                print(f"\n文本: {test_data['sentence'][idx]}")
                print(f"真实标签: {'积极' if labels[idx] == 1 else '消极'}")
                print(f"预测标签: {'积极' if preds[idx] == 1 else '消极'}")
            elif self.task_name == "qqp":
                print(f"\n问题1: {test_data['question1'][idx]}")
                print(f"问题2: {test_data['question2'][idx]}")
                print(f"真实等价: {'是' if labels[idx] == 1 else '否'}")
                print(f"预测等价: {'是' if preds[idx] == 1 else '否'}")

        print("\n错误分析完成，展示前5个错误样本")