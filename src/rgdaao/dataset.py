from __future__ import annotations


class ProteinRegressionDataset:
    """Sequence/label adapter with no silent sequence truncation."""
    def __init__(self, frame, tokenizer, max_length=512):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if (self.frame.mutated_sequence.str.len() + tokenizer.num_special_tokens_to_add(False) > max_length).any():
            raise ValueError('Sequence exceeds maximum token length; truncation is prohibited')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        import torch
        row = self.frame.iloc[index]
        encoded = self.tokenizer(row['mutated_sequence'], truncation=False, return_tensors='pt')
        if encoded['input_ids'].shape[1] > self.max_length:
            raise ValueError('Tokenized sequence exceeds maximum length')
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(float(row['activity']), dtype=torch.float32)
        return item
