from __future__ import annotations


def load_esm2(model_name: str = "facebook/esm2_t6_8M_UR50D"):
    """Load a compact ESM-2 model/tokenizer lazily.

    The dependency is optional so data/metric tests remain lightweight.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'Install ML dependencies with: pip install -e ".[ml]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    return tokenizer, model


def freeze_backbone(model) -> None:
    """Freeze the ESM backbone while leaving the regression head trainable."""
    base = getattr(model, "esm", None)
    if base is None:
        raise ValueError("Expected a Hugging Face ESM sequence-classification model")
    for parameter in base.parameters():
        parameter.requires_grad = False
