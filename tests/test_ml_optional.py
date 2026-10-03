"""Optional offline ML checks using a tiny random ESM; no pretrained downloads."""
import json
from pathlib import Path
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('accelerate')
from transformers import EsmConfig, EsmForSequenceClassification, EsmTokenizer
from rgdaao.config import load_config
from rgdaao.dataset import ProteinRegressionDataset
from rgdaao.model import load_regression_model
from rgdaao.training import file_sha256, write_json


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.set_num_threads(2)
    path = tmp_path / 'tiny'
    path.mkdir()
    tokens = ['<cls>', '<pad>', '<eos>', '<unk>'] + list('ACDEFGHIKLMNPQRSTVWY') + ['<mask>']
    vocab = path / 'vocab.txt'
    vocab.write_text('\n'.join(tokens) + '\n')
    tokenizer = EsmTokenizer(str(vocab))
    tokenizer.save_pretrained(path)
    config = EsmConfig(vocab_size=len(tokens), hidden_size=16, num_hidden_layers=1,
                       num_attention_heads=2, intermediate_size=32, max_position_embeddings=512,
                       pad_token_id=1, mask_token_id=len(tokens)-1, num_labels=1, problem_type='regression')
    EsmForSequenceClassification(config).save_pretrained(path)
    return path


def test_no_truncation(tiny_checkpoint):
    tokenizer, _ = load_regression_model(checkpoint=tiny_checkpoint)
    with pytest.raises(ValueError, match='truncation'):
        ProteinRegressionDataset(pd.DataFrame({'mutated_sequence': ['A'*600], 'activity': [0]}), tokenizer)


@pytest.mark.parametrize('training_seed', [42, 43])
def test_train_select_serialize_then_evaluate(tmp_path, tiny_checkpoint, monkeypatch, training_seed):
    # Synthetic fixtures exercise control flow, not performance of ESM-2.
    import rgdaao.train as training
    from rgdaao.evaluate import evaluate
    original = load_regression_model
    monkeypatch.setattr(training, 'load_regression_model',
                        lambda name, frozen, revision: original(checkpoint=tiny_checkpoint, freeze_backbone=frozen))
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'wt.fasta').write_text('>synthetic\nACDE\n')
    frames = {
        'train': pd.DataFrame({'mutation': ['A1V', 'C2W', 'D3F', 'E4A'],
                               'mutated_sequence': ['VCDE', 'AWDE', 'ACFE', 'ACDA'], 'activity': [0.1, -0.2, 0.3, -0.1]}),
        'val': pd.DataFrame({'mutation': ['A1L', 'C2V'], 'mutated_sequence': ['LCDE', 'AVDE'], 'activity': [0.2, -0.1]}),
        'test': pd.DataFrame({'mutation': ['D3W', 'E4L'], 'mutated_sequence': ['ACWE', 'ACDL'], 'activity': [0.1, 0.4]}),
    }
    for name, frame in frames.items():
        frame.to_csv(data / f'{name}.csv', index=False)
    manifest = {'seed': 42, 'dataset_sha256': 'synthetic-fixture',
                'split_sha256': {n: file_sha256(data / f'{n}.csv') for n in frames}}
    if training_seed != 42:
        manifest['training_seeds'] = [42, 43, 44]
    write_json(data / 'split_summary.json', manifest)
    write_json(data / 'provenance.json', {'dataset_sha256': 'synthetic-fixture',
                                         'counts': {'matches_paper_count': True}})
    test_bytes = (data / 'test.csv').read_bytes()
    (data / 'test.csv').unlink()  # training MUST succeed with no accessible test set
    config = load_config('configs/esm2_frozen.yaml')
    config['training']['seed'] = training_seed
    config['training'].update(epochs=2, batch_size=2)
    output = tmp_path / 'experiment'
    training.train(config, data, output)
    run = json.loads((output / 'run.json').read_text())
    history = json.loads((output / 'history.json').read_text())
    assert run['best_validation_spearman'] == max(h['eval_spearman'] for h in history if 'eval_spearman' in h)
    assert run['status'] == 'trained' and not (output / 'predictions.csv').exists()
    with pytest.raises(FileExistsError):
        training.train(config, data, output)
    (data / 'test.csv').write_bytes(test_bytes)
    evaluate(output, data)
    assert json.loads((output / 'run.json').read_text())['status'] == 'evaluated'
    assert pd.read_csv(output / 'predictions.csv').mutation.tolist() == ['D3W', 'E4L']
    with pytest.raises(ValueError, match='previous test evaluation'):
        evaluate(output, data)
