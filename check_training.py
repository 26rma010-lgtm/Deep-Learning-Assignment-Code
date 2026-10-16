
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import Seq2Seq, count_parameters


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"


def main():
    print("=" * 50)
    print("TRAINING COMPATIBILITY CHECK")
    print("=" * 50)

    with open(DATA_DIR / "source_vocab.json", encoding="utf-8") as f:
        source_vocab = json.load(f)

    with open(DATA_DIR / "target_vocab.json", encoding="utf-8") as f:
        target_vocab = json.load(f)

    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)

    source_vocab_size = len(source_vocab["tokens"])
    target_vocab_size = len(target_vocab["tokens"])
    pad_id = target_vocab["token_to_id"]["<pad>"]

    # Use two real training examples.
    sources = train_data["source"][:2]
    targets = train_data["target"][:2]

    source_lengths = torch.tensor(
        [len(sequence) for sequence in sources],
        dtype=torch.long,
    )

    sources = nn.utils.rnn.pad_sequence(
        [torch.tensor(x, dtype=torch.long) for x in sources],
        batch_first=True,
        padding_value=0,
    )

    targets = nn.utils.rnn.pad_sequence(
        [torch.tensor(x, dtype=torch.long) for x in targets],
        batch_first=True,
        padding_value=pad_id,
    )

    decoder_inputs = targets[:, :-1]
    labels = targets[:, 1:]

    model = Seq2Seq(
        input_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=1,
        cell_type="GRU",
        dropout=0.2,
        padding_idx=0,
    )

    logits = model(
        sources,
        decoder_inputs,
        source_lengths=source_lengths,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    loss = criterion(
        logits.reshape(-1, target_vocab_size),
        labels.reshape(-1),
    )

    loss.backward()

    print("Source batch shape:", tuple(sources.shape))
    print("Decoder input shape:", tuple(decoder_inputs.shape))
    print("Label shape:", tuple(labels.shape))
    print("Output shape:", tuple(logits.shape))
    print("Initial loss:", round(loss.item(), 4))
    print("Trainable parameters:", count_parameters(model))
    print("Backward pass: successful")

    assert torch.isfinite(loss), "Loss is not finite."
    assert logits.shape[:2] == labels.shape
    assert logits.shape[-1] == target_vocab_size

    print("\nPASS: Real data works with the model and loss.")


if __name__ == "__main__":
    main()
