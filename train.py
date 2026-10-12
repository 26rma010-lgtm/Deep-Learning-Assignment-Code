
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader

from model import Seq2Seq


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
)

TRAIN_FILE = DATA_DIR / "hi.translit.sampled.train.tsv"
DEV_FILE = DATA_DIR / "hi.translit.sampled.dev.tsv"

CELL_TYPES = ["RNN", "GRU", "LSTM"]

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128

ENCODER_LAYERS = 1
DECODER_LAYERS = 1
DROPOUT = 0.0

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

PATIENCE = 4

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


# ============================================================
# 2. RANDOM SEED
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_pairs(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        keep_default_na=False
    )

    # Dakshina column 0 = Hindi
    # Dakshina column 1 = Romanized Hindi
    # Input is Romanized Hindi; target is Hindi.
    pairs = [
        (
            str(row.iloc[1]).strip(),
            str(row.iloc[0]).strip()
        )
        for _, row in df.iterrows()
        if str(row.iloc[0]).strip()
        and str(row.iloc[1]).strip()
    ]

    return pairs


# ============================================================
# 4. BUILD CHARACTER VOCABULARIES
# ============================================================

def build_vocab(pairs, idx):
    chars = sorted({
        char
        for pair in pairs
        for char in pair[idx]
    })

    vocab = {
        PAD: 0,
        SOS: 1,
        EOS: 2,
        UNK: 3
    }

    for char in chars:
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


# ============================================================
# 5. ENCODE CHARACTERS
# ============================================================

def encode(text, vocab, add_sos=False):
    ids = []

    if add_sos:
        ids.append(vocab[SOS])

    ids.extend([
        vocab.get(char, vocab[UNK])
        for char in text
    ])

    ids.append(vocab[EOS])

    return ids


# ============================================================
# 6. DATASET CLASS
# ============================================================

class PairDataset(Dataset):

    def __init__(self, pairs, source_vocab, target_vocab):
        self.pairs = pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        roman, hindi = self.pairs[index]

        source = torch.tensor(
            encode(roman, self.source_vocab),
            dtype=torch.long
        )

        target = torch.tensor(
            encode(
                hindi,
                self.target_vocab,
                add_sos=True
            ),
            dtype=torch.long
        )

        return source, target


# ============================================================
# 7. PAD BATCHES
# ============================================================

def make_collate(source_vocab, target_vocab):

    def collate(batch):
        sources, targets = zip(*batch)

        lengths = torch.tensor(
            [len(source) for source in sources],
            dtype=torch.long
        )

        max_source = max(len(source) for source in sources)
        max_target = max(len(target) for target in targets)

        sources = [
            torch.cat((
                source,
                torch.full(
                    (max_source - len(source),),
                    source_vocab[PAD],
                    dtype=torch.long
                )
            ))
            for source in sources
        ]

        targets = [
            torch.cat((
                target,
                torch.full(
                    (max_target - len(target),),
                    target_vocab[PAD],
                    dtype=torch.long
                )
            ))
            for target in targets
        ]

        return (
            torch.stack(sources),
            torch.stack(targets),
            lengths
        )

    return collate


# ============================================================
# 8. RUN ONE TRAINING OR VALIDATION EPOCH
# ============================================================

def epoch_run(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for source, target, lengths in loader:
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        decoder_input = target[:, :-1]
        expected = target[:, 1:]

        with torch.set_grad_enabled(training):
            logits = model(
                source,
                decoder_input,
                lengths
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1)
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

        # Ignore padding tokens when measuring accuracy.
        mask = expected.ne(0)
        token_count = int(mask.sum().item())

        total_loss += loss.item() * token_count
        total_tokens += token_count

        predictions = logits.argmax(dim=-1)

        total_correct += int(
            ((predictions == expected) & mask).sum().item()
        )

    average_loss = total_loss / max(total_tokens, 1)

    character_accuracy = (
        100 * total_correct / max(total_tokens, 1)
    )

    return average_loss, character_accuracy


# ============================================================
# 9. TRAIN MODEL
# ============================================================

def train_model(
    kind,
    train_pairs,
    dev_pairs,
    source_vocab,
    target_vocab,
    embedding_size=None,
    hidden_size=None,
    batch_size=None,
    learning_rate=None,
    encoder_layers=None,
    decoder_layers=None,
    epochs=None,
    run=None,
    dropout=None
):
    # Use defaults for baseline training.
    # W&B sweep values override these defaults.

    embedding_size = (
        EMBEDDING_SIZE if embedding_size is None
        else int(embedding_size)
    )

    hidden_size = (
        HIDDEN_SIZE if hidden_size is None
        else int(hidden_size)
    )

    batch_size = (
        BATCH_SIZE if batch_size is None
        else int(batch_size)
    )

    learning_rate = (
        LEARNING_RATE if learning_rate is None
        else float(learning_rate)
    )

    encoder_layers = (
        ENCODER_LAYERS if encoder_layers is None
        else int(encoder_layers)
    )

    decoder_layers = (
        DECODER_LAYERS if decoder_layers is None
        else int(decoder_layers)
    )

    epochs = (
        EPOCHS if epochs is None
        else int(epochs)
    )

    dropout = (
        DROPOUT if dropout is None
        else float(dropout)
    )

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be between 0.0 and 1.0")

    collate = make_collate(source_vocab, target_vocab)

    train_loader = DataLoader(
        PairDataset(train_pairs, source_vocab, target_vocab),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    dev_loader = DataLoader(
        PairDataset(dev_pairs, source_vocab, target_vocab),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    # Create the configurable encoder-decoder model.
    model = Seq2Seq(
        input_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        input_pad_id=source_vocab[PAD],
        embedding_size=embedding_size,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        cell_type=kind,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        dropout=dropout
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=target_vocab[PAD]
    )

    best_loss = float("inf")
    best_accuracy = 0.0
    stale_epochs = 0

    history = []
    start_time = time.time()

    # Unique names prevent sweep runs overwriting each other.
    run_name = run.id if run is not None else kind.lower()

    checkpoint_path = (
        PROJECT_DIR / f"best_{run_name}_model.pt"
    )

    # Training loop.
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_accuracy = epoch_run(
            model,
            train_loader,
            criterion,
            optimizer
        )

        dev_loss, dev_accuracy = epoch_run(
            model,
            dev_loader,
            criterion
        )

        epoch_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_character_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_character_accuracy": dev_accuracy,
            "epoch_seconds": epoch_seconds
        }

        history.append(row)

        print(
            f"{kind} | Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Char Accuracy: {train_accuracy:.2f}% | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Char Accuracy: {dev_accuracy:.2f}%"
        )

        # Log metrics and hyperparameters to W&B.
        if run is not None:
            wandb.log({
                **row,
                "trainable_parameters": parameter_count,
                "cell_type": kind,
                "embedding_size": embedding_size,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "encoder_layers": encoder_layers,
                "decoder_layers": decoder_layers,
                "dropout": dropout
            })

        # Save the best checkpoint based on validation loss.
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_accuracy = dev_accuracy
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "source_vocab": source_vocab,
                    "target_vocab": target_vocab,
                    "cell_type": kind,
                    "embedding_size": embedding_size,
                    "hidden_size": hidden_size,
                    "encoder_layers": encoder_layers,
                    "decoder_layers": decoder_layers,
                    "dropout": dropout,
                    "trainable_parameters": parameter_count,
                    "best_dev_loss": best_loss,
                    "best_dev_character_accuracy": best_accuracy
                },
                checkpoint_path
            )

        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(
                f"Early stopping after {PATIENCE} "
                "epochs without validation-loss improvement."
            )
            break

    # Save per-epoch history.
    history_path = PROJECT_DIR / f"history_{run_name}.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    return {
        "model": kind,
        "trainable_parameters": parameter_count,
        "best_dev_loss": best_loss,
        "best_dev_character_accuracy": best_accuracy,
        "epochs_run": len(history),
        "checkpoint": checkpoint_path.name,
        "history_file": history_path.name,
        "training_minutes": (time.time() - start_time) / 60
    }


# ============================================================
# 10. LOAD TRAINING AND VALIDATION DATA
# ============================================================

def load_training_data():
    train_pairs = load_pairs(TRAIN_FILE)
    dev_pairs = load_pairs(DEV_FILE)

    # Build vocabularies from training data only.
    # Source vocabulary is Romanized Hindi (pair index 0).
    # Target vocabulary is Hindi (pair index 1).
    source_vocab = build_vocab(train_pairs, 0)
    target_vocab = build_vocab(train_pairs, 1)

    print(f"Device: {DEVICE}")
    print(f"Training examples: {len(train_pairs)}")
    print(f"Validation examples: {len(dev_pairs)}")
    print(f"Source vocabulary size: {len(source_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")

    return (
        train_pairs,
        dev_pairs,
        source_vocab,
        target_vocab
    )


# ============================================================
# 11. BASELINE TRAINING: RNN, GRU, LSTM
# ============================================================

def main():
    seed_everything()

    (
        train_pairs,
        dev_pairs,
        source_vocab,
        target_vocab
    ) = load_training_data()

    summaries = []

    for kind in CELL_TYPES:
        seed_everything()

        result = train_model(
            kind,
            train_pairs,
            dev_pairs,
            source_vocab,
            target_vocab
        )

        summaries.append(result)

    summary_path = PROJECT_DIR / "training_summary.csv"

    pd.DataFrame(summaries).to_csv(
        summary_path,
        index=False
    )

    print("Training summary saved to:", summary_path)


# ============================================================
# 12. WANDB SWEEP TRAINING FUNCTION
# ============================================================

def sweep_train():
    with wandb.init(project="hindi-transliteration") as run:
        config = wandb.config

        seed_everything()

        (
            train_pairs,
            dev_pairs,
            source_vocab,
            target_vocab
        ) = load_training_data()

        result = train_model(
            kind=config.cell_type,
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            epochs=config.epochs,
            dropout=getattr(config, "dropout", DROPOUT),
            run=run
        )

        # Store best validation metrics in the W&B summary.
        wandb.summary["best_dev_loss"] = (
            result["best_dev_loss"]
        )

        wandb.summary["best_dev_character_accuracy"] = (
            result["best_dev_character_accuracy"]
        )

        # The sweep metric should use the best validation accuracy.
        wandb.summary["dev_character_accuracy"] = (
            result["best_dev_character_accuracy"]
        )

        wandb.summary["trainable_parameters"] = (
            result["trainable_parameters"]
        )

        wandb.summary["epochs_run"] = (
            result["epochs_run"]
        )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
