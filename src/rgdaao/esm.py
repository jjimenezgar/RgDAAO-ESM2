"""Backward-compatible entry points; model construction lives in model.py."""
from .model import load_regression_model


def load_esm2(model_name='facebook/esm2_t6_8M_UR50D'):
    return load_regression_model(model_name)


def freeze_backbone(model):
    if not hasattr(model, 'esm'):
        raise ValueError('Expected an ESM regression model')
    for parameter in model.esm.parameters():
        parameter.requires_grad = False
    model.backbone_frozen = True
    model.esm.eval()
