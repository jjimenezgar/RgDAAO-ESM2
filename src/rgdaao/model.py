from __future__ import annotations
from .config import MODEL_NAME, MODEL_REVISION


def load_regression_model(model_name=MODEL_NAME, freeze_backbone=False, revision=MODEL_REVISION,
                          checkpoint=None):
    """The same Hugging Face CLS regression head in both modes."""
    from transformers import EsmForSequenceClassification, AutoTokenizer

    class ActivityRegressor(EsmForSequenceClassification):
        def train(self, mode=True):
            super().train(mode)
            if getattr(self, 'backbone_frozen', False):
                self.esm.eval()  # no stochastic backbone dropout in frozen mode
            return self

    source = str(checkpoint) if checkpoint else model_name
    kwargs = {} if checkpoint else {'revision': revision}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = ActivityRegressor.from_pretrained(
        source, num_labels=1, problem_type='regression', **kwargs)
    model.backbone_frozen = freeze_backbone
    for parameter in model.esm.parameters():
        parameter.requires_grad = not freeze_backbone
    model.train()
    return tokenizer, model


def trainable_parameters(model) -> tuple[int, int]:
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def make_optimizer(model, config):
    """Same head LR in both modes; smaller LR for the pretrained backbone."""
    import torch
    groups = [{'params': [p for p in model.classifier.parameters() if p.requires_grad],
               'lr': config['head_learning_rate']}]
    backbone = [p for p in model.esm.parameters() if p.requires_grad]
    if backbone:
        groups.append({'params': backbone, 'lr': config['learning_rate']})
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
