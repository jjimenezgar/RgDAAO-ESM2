from __future__ import annotations


class ProteinRegressionDataset:
    """Minimal torch Dataset for protein-sequence regression."""

    def __init__(self, frame, tokenizer, max_length: int = 512):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        import torch

        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["mutated_sequence"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(float(row["activity"]), dtype=torch.float32)
        return item
