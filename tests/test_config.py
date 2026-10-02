from copy import deepcopy
import pytest
from rgdaao.config import load_config, validate_config, check_comparison


def test_paired_configs():
    a, b = load_config('configs/esm2_frozen.yaml'), load_config('configs/esm2_finetune.yaml')
    check_comparison(a, b)
    b['training']['seed'] = 7
    with pytest.raises(ValueError, match='share'):
        check_comparison(a, b)


@pytest.mark.parametrize('section,key,value', [
    ('model', 'freeze_backbone', 'false'), ('training', 'seed', -1),
    ('training', 'seed', True), ('training', 'epochs', 0), ('training', 'batch_size', 0),
    ('training', 'learning_rate', float('nan')), ('training', 'head_learning_rate', 0)])
def test_invalid_config(section, key, value):
    cfg = deepcopy(load_config('configs/esm2_frozen.yaml'))
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_wrong_target_and_unknown_keys():
    cfg = load_config('configs/esm2_frozen.yaml')
    cfg['target'] = 'expression'
    with pytest.raises(ValueError, match='activity'):
        validate_config(cfg)
    cfg['target'] = 'activity'
    cfg['training']['test_for_selection'] = True
    with pytest.raises(ValueError):
        validate_config(cfg)
