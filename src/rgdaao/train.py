from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .dataset import ProteinRegressionDataset
from .metrics import regression_metrics
from .model import load_regression_model, trainable_parameters
from .training import save_evaluation, set_seed


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)
    return regression_metrics(labels, predictions)


def main() -> None:
    p = argparse.ArgumentParser(description="Train an ESM-2 activity regressor.")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["training"]["seed"])
    set_seed(seed)

    train_df = pd.read_csv(Path(args.data_dir) / "train.csv")
    val_df = pd.read_csv(Path(args.data_dir) / "val.csv")
    test_df = pd.read_csv(Path(args.data_dir) / "test.csv")

    tokenizer, model = load_regression_model(
        config["model"]["name"],
        bool(config["model"]["freeze_backbone"]),
    )

    try:
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError('Install ML dependencies with: pip install -e ".[ml]"') from exc

    trainable, total = trainable_parameters(model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_size.json").write_text(json.dumps({
        "trainable_parameters": trainable,
        "total_parameters": total,
    }, indent=2) + "\n")

    train_ds = ProteinRegressionDataset(train_df, tokenizer)
    val_ds = ProteinRegressionDataset(val_df, tokenizer)
    test_ds = ProteinRegressionDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=int(config["training"]["batch_size"]),
        per_device_eval_batch_size=int(config["training"]["batch_size"]),
        num_train_epochs=int(config["training"]["epochs"]),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # The held-out test set is evaluated only after model selection on validation.
    prediction = trainer.predict(test_ds)
    y_pred = prediction.predictions.reshape(-1)
    y_true = test_df["activity"].to_numpy()
    mutations = (
        test_df["mutation"].astype(str).tolist()
        if "mutation" in test_df
        else [str(i) for i in range(len(test_df))]
    )
    metrics = save_evaluation(mutations, y_true, y_pred, output)
    tokenizer.save_pretrained(output / "best_model")
    trainer.save_model(output / "best_model")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
