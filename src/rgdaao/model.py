from __future__ import annotations


def load_regression_model(model_name: str, freeze_backbone: bool):
    """Load ESM-2 for sequence-level regression."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError('Install ML dependencies with: pip install -e ".[ml]"') from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    if freeze_backbone:
        for parameter in model.esm.parameters():
            parameter.requires_grad = False
    return tokenizer, model


def trainable_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
