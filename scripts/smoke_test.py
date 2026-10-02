"""Real pretrained ESM-2 optimizer/serialization checks, NOT a benchmark."""
from __future__ import annotations
import argparse
import gc
import tempfile
from pathlib import Path
import torch
from transformers import DataCollatorWithPadding
from rgdaao.config import load_config, MODEL_NAME, MODEL_REVISION
from rgdaao.data import load_variants
from rgdaao.dataset import ProteinRegressionDataset
from rgdaao.model import load_regression_model, trainable_parameters, make_optimizer
from rgdaao.source import read_fasta
from rgdaao.training import set_seed, write_json, environment, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output', type=Path, default=Path('results/smoke/smoke.json'))
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # Training-only subset: test and validation files are never read.
    frame = load_variants(args.data_dir / 'train.csv', read_fasta(args.data_dir / 'wt.fasta')).head(2)
    report = {'purpose': 'software smoke test, not benchmark performance', 'rows': len(frame),
              'source': 'train.csv only', 'sequence_length': int(frame.mutated_sequence.str.len().iloc[0]),
              'environment': environment(), 'modes': {},
              'model_name': MODEL_NAME, 'model_revision': MODEL_REVISION,
              'train_csv_sha256': file_sha256(args.data_dir / 'train.csv'),
              'source_sha256': {str(p): file_sha256(p) for p in
                  sorted(Path('src/rgdaao').glob('*.py')) + sorted(Path('configs').glob('*.yaml'))
                  + [Path('scripts/smoke_test.py')]}}
    initial_head = None
    for mode in ('frozen', 'finetune'):
        config = load_config(f'configs/esm2_{mode}.yaml')
        set_seed(config['training']['seed'])
        tokenizer, model = load_regression_model(freeze_backbone=config['model']['freeze_backbone'])
        ds = ProteinRegressionDataset(frame, tokenizer)
        batch = DataCollatorWithPadding(tokenizer)([ds[i] for i in range(len(ds))])
        assert batch['input_ids'].shape == (len(frame), 367)
        assert not torch.any(batch['input_ids'] == tokenizer.unk_token_id)
        before_head = {n: v.detach().clone() for n, v in model.classifier.named_parameters()}
        if initial_head is None:
            initial_head = before_head
        else:
            assert all(torch.equal(v, initial_head[n]) for n, v in before_head.items())
        before_base = {n: v.detach().clone() for n, v in model.esm.named_parameters()}
        model.train()
        assert model.esm.training == (mode == 'finetune')
        optimizer = make_optimizer(model, config['training'])
        optimizer.zero_grad()
        output = model(**batch)
        assert output.logits.shape == (len(frame), 1) and torch.isfinite(output.loss)
        output.loss.backward()
        head_grad = any(v.grad is not None and v.grad.abs().sum() > 0 for v in model.classifier.parameters())
        base_grad = any(v.grad is not None and v.grad.abs().sum() > 0 for v in model.esm.parameters())
        assert head_grad and base_grad == (mode == 'finetune')
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        changed_head = any(not torch.equal(v, before_head[n]) for n, v in model.classifier.named_parameters())
        changed_base = any(not torch.equal(v, before_base[n]) for n, v in model.esm.named_parameters())
        assert changed_head and changed_base == (mode == 'finetune')
        model.eval()
        with torch.no_grad():
            expected = model(**batch).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            _, restored = load_regression_model(checkpoint=directory, freeze_backbone=mode == 'frozen')
            restored.eval()
            with torch.no_grad():
                assert torch.allclose(restored(**batch).logits, expected, atol=1e-7, rtol=1e-6)
        trainable, total = trainable_parameters(model)
        report['modes'][mode] = {'passed': True, 'trainable_parameters': trainable, 'total_parameters': total,
                                 'head_updated': changed_head, 'backbone_updated': changed_base,
                                 'checkpoint_roundtrip': True, 'finite_loss_and_gradients': True}
        print(f'{mode}: forward, loss, backward, optimizer and checkpoint reload PASSED', flush=True)
        del model, restored, optimizer, output, before_base
        gc.collect()
    report['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
