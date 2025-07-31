import argparse
from preprocessing import SparkDataPreprocessor
from modeling import GLUEClassifierTrainer
from evaluation import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="GLUE文本分类实验")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "qqp"], help="GLUE任务名称")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="预训练模型名称 (e.g., bert-base-uncased, roberta-base)")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--spark_master", type=str, default="local[*]", help="Spark Master URL")
    args = parser.parse_args()

    # 1. Spark数据预处理
    print("=" * 50)
    print(f"开始Spark数据预处理: {args.task}")
    preprocessor = SparkDataPreprocessor(
        task_name=args.task,
        spark_master=args.spark_master
    )
    train_dataset, val_dataset, test_dataset = preprocessor.process_data()

    # 2. 模型训练
    print("\n" + "=" * 50)
    print(f"开始模型训练: {args.model_name} on {args.task}")
    trainer = GLUEClassifierTrainer(
        model_name=args.model_name,
        task_name=args.task,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    model, tokenizer = trainer.train(train_dataset, val_dataset)

    # 3. 模型评估
    print("\n" + "=" * 50)
    print(f"开始模型评估")
    evaluator = ModelEvaluator(model, tokenizer, args.task, args.batch_size)
    test_results = evaluator.evaluate(test_dataset)

    # 4. 资源监控报告
    print("\n" + "=" * 50)
    print("资源使用报告:")
    print(f"- Spark预处理时间: {preprocessor.preprocess_time:.2f}秒")
    print(f"- 模型训练时间: {trainer.training_time:.2f}秒")
    print(f"- 峰值GPU显存使用: {trainer.max_gpu_memory:.2f}MB")

    # 5. 结果保存
    print("\n" + "=" * 50)
    print("保存最终结果...")
    model.save_pretrained(f"results/{args.task}_{args.model_name}_model")
    tokenizer.save_pretrained(f"results/{args.task}_{args.model_name}_tokenizer")

    print("\n实验完成!")


if __name__ == "__main__":
    main()