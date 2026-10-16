
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from model import Seq2Seq, count_parameters


# ------------------------------------------------------------
# Paths and settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints" / "wandb_runs"
RESULTS_DIR = ROOT / "results" / "wandb_runs"

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 4
GRAD_CLIP = 1.0

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

WANDB_PROJECT = "hindi-transliteration"

DEFAULT_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "cell_type": "GRU",
    "dropout": 0.2,
    "learning_rate": 0.001,
}


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class TransliterationDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.sources = data["source"]
        self.targets = data["target"]

        if len(self.sources) != len(self.targets):
            raise ValueError("Source/target count mismatch.")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sources[index], dtype=torch.long),
            torch.tensor(self.targets[index], dtype=torch.long),
        )


def collate_batch(batch):
    sources, targets = zip(*batch)

    source_lengths = torch.tensor(
        [len(source) for source in sources],
        dtype=torch.long,
    )

    sources = nn.utils.rnn.pad_sequence(
        sources,
        batch_first=True,
        padding_value=0,
    )

    targets = nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=0,
    )

    # Teacher forcing: input excludes the final target token.
    decoder_inputs = targets[:, :-1]
    labels = targets[:, 1:]

    return sources, source_lengths, decoder_inputs, labels


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def evaluate(model, loader, criterion, device, pad_id):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_characters = 0
    exact_words = 0
    total_words = 0

    with torch.no_grad():
        for sources, source_lengths, decoder_inputs, labels in loader:
            sources = sources.to(device)
            source_lengths = source_lengths.to(device)
            decoder_inputs = decoder_inputs.to(device)
            labels = labels.to(device)

            logits = model(
                sources,
                decoder_inputs,
                source_lengths=source_lengths,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            predictions = logits.argmax(dim=-1)
            valid_mask = labels.ne(pad_id)

            valid_tokens = valid_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            correct_characters += (
                predictions.eq(labels) & valid_mask
            ).sum().item()

            # Exact match under teacher forcing.
            exact_per_word = (
                predictions.eq(labels) | ~valid_mask
            ).all(dim=1)

            exact_words += exact_per_word.sum().item()
            total_words += labels.size(0)

    return {
        "dev_loss": total_loss / max(total_tokens, 1),
        "dev_character_accuracy": (
            100.0 * correct_characters / max(total_tokens, 1)
        ),
        "dev_word_accuracy": (
            100.0 * exact_words / max(total_words, 1)
        ),
    }


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------

def main():
    with wandb.init(
        project=WANDB_PROJECT,
        config=DEFAULT_CONFIG,
    ) as run:

        config = dict(wandb.config)
        set_seed(SEED)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", device)
        print("W&B run:", run.name)
        print("Run ID:", run.id)
        print("Configuration:", config)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load vocabularies.
        with open(
            DATA_DIR / "source_vocab.json",
            "r",
            encoding="utf-8",
        ) as file:
            source_vocab = json.load(file)

        with open(
            DATA_DIR / "target_vocab.json",
            "r",
            encoding="utf-8",
        ) as file:
            target_vocab = json.load(file)

        source_vocab_size = len(source_vocab["tokens"])
        target_vocab_size = len(target_vocab["tokens"])
        pad_id = target_vocab["token_to_id"][PAD_TOKEN]

        # Load standard train and development splits.
        train_dataset = TransliterationDataset(
            DATA_DIR / "train.json"
        )
        dev_dataset = TransliterationDataset(
            DATA_DIR / "dev.json"
        )

        generator = torch.Generator().manual_seed(SEED)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
            generator=generator,
        )

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
        )

        # Build basic encoder-decoder model.
        model = Seq2Seq(
            input_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            encoder_layers=int(config["encoder_layers"]),
            decoder_layers=int(config["decoder_layers"]),
            cell_type=str(config["cell_type"]),
            dropout=float(config["dropout"]),
            padding_idx=0,
        ).to(device)

        parameter_count = count_parameters(model)

        print("Training examples:", len(train_dataset))
        print("Development examples:", len(dev_dataset))
        print("Source vocabulary:", source_vocab_size)
        print("Target vocabulary:", target_vocab_size)
        print("Architecture: Basic Seq2Seq without attention")
        print("Cell type:", config["cell_type"])
        print("Trainable parameters:", parameter_count)

        wandb.config.update(
            {
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "seed": SEED,
                "train_examples": len(train_dataset),
                "dev_examples": len(dev_dataset),
                "source_vocab_size": source_vocab_size,
                "target_vocab_size": target_vocab_size,
                "trainable_parameters": parameter_count,
                "attention": False,
            },
            allow_val_change=True,
        )

        criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
        )

        best_dev_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        checkpoint_path = (
            CHECKPOINT_DIR / f"{run.id}_best_seq2seq.pt"
        )
        history_path = (
            RESULTS_DIR / f"{run.id}_history.json"
        )

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()

            total_loss = 0.0
            total_tokens = 0

            for (
                sources,
                source_lengths,
                decoder_inputs,
                labels,
            ) in train_loader:

                sources = sources.to(device)
                source_lengths = source_lengths.to(device)
                decoder_inputs = decoder_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(
                    sources,
                    decoder_inputs,
                    source_lengths=source_lengths,
                )

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

                valid_tokens = labels.ne(pad_id).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

            train_loss = total_loss / max(total_tokens, 1)

            dev_metrics = evaluate(
                model,
                dev_loader,
                criterion,
                device,
                pad_id,
            )

            dev_loss = dev_metrics["dev_loss"]

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **dev_metrics,
            }
            history.append(record)

            # Log metrics to W&B.
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                    "dev_character_accuracy":
                        dev_metrics["dev_character_accuracy"],
                    "dev_word_accuracy":
                        dev_metrics["dev_word_accuracy"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Dev Loss: {dev_loss:.4f} | "
                f"Character Accuracy: "
                f"{dev_metrics['dev_character_accuracy']:.2f}% | "
                f"Word Accuracy: "
                f"{dev_metrics['dev_word_accuracy']:.2f}%"
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_loss": best_dev_loss,
                    "config": {
                        "embedding_dim": int(config["embedding_dim"]),
                        "hidden_dim": int(config["hidden_dim"]),
                        "encoder_layers": int(config["encoder_layers"]),
                        "decoder_layers": int(config["decoder_layers"]),
                        "cell_type": str(config["cell_type"]),
                        "dropout": float(config["dropout"]),
                        "learning_rate": float(config["learning_rate"]),
                    },
                }

                torch.save(checkpoint, checkpoint_path)
                print("Saved best checkpoint.")
            else:
                epochs_without_improvement += 1

            with open(
                history_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(history, file, indent=2)

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

        wandb.summary["best_dev_loss"] = best_dev_loss
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["checkpoint_path"] = str(checkpoint_path)

        print("\nTraining completed.")
        print("Best development loss:", best_dev_loss)
        print("Best epoch:", best_epoch)
        print("Checkpoint:", checkpoint_path)
        print("History:", history_path)


if __name__ == "__main__":
    main()
