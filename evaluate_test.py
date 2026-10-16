
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from model import Seq2Seq


# ============================================================
# 1. Paths and device
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "wandb_runs"
    / "u7qnruwi_best_seq2seq.pt"
)

TEST_PATH = DATA_DIR / "test.json"
SOURCE_VOCAB_PATH = DATA_DIR / "source_vocab.json"
TARGET_VOCAB_PATH = DATA_DIR / "target_vocab.json"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "test_evaluation.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PAD_ID = 0
SOS_ID = 2
EOS_ID = 3

BATCH_SIZE = 64
MAX_OUTPUT_LENGTH = 100


# ============================================================
# 2. Load JSON files
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


test_json = load_json(TEST_PATH)
source_vocab = load_json(SOURCE_VOCAB_PATH)
target_vocab = load_json(TARGET_VOCAB_PATH)

source_tokens = source_vocab["tokens"]
target_tokens = target_vocab["tokens"]

target_token_to_id = target_vocab["token_to_id"]

id_to_target = {
    int(token_id): token
    for token, token_id in target_token_to_id.items()
}

# The test file contains parallel source and target lists.
test_sources = test_json["source"]
test_targets = test_json["target"]

if len(test_sources) != len(test_targets):
    raise ValueError(
        "The numbers of source and target samples do not match."
    )

if not test_sources:
    raise ValueError("The test dataset is empty.")

test_data = list(zip(test_sources, test_targets))


# ============================================================
# 3. Load the saved checkpoint and its configuration
# ============================================================
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

config = checkpoint["config"]

print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print("Device:", DEVICE)
print("Test examples:", len(test_data))
print("Source vocabulary:", len(source_tokens))
print("Target vocabulary:", len(target_tokens))
print("Checkpoint epoch:", checkpoint["epoch"])
print("Best development loss:", checkpoint["best_dev_loss"])
print("Saved model configuration:", config)


# Use the saved configuration rather than hard-coding
# potentially different hyperparameters.
model = Seq2Seq(
    input_vocab_size=len(source_tokens),
    target_vocab_size=len(target_tokens),
    embedding_dim=int(config["embedding_dim"]),
    hidden_dim=int(config["hidden_dim"]),
    encoder_layers=int(config["encoder_layers"]),
    decoder_layers=int(config["decoder_layers"]),
    cell_type=config["cell_type"],
    dropout=float(config["dropout"]),
    padding_idx=PAD_ID,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nCheckpoint loaded successfully.")
print("Model cell type:", config["cell_type"])


# ============================================================
# 4. Prepare a batch with padding
# ============================================================
def prepare_batch(samples):
    sources = [
        torch.tensor(source, dtype=torch.long)
        for source, _ in samples
    ]

    source_lengths = torch.tensor(
        [len(source) for source in sources],
        dtype=torch.long,
    )

    padded_sources = pad_sequence(
        sources,
        batch_first=True,
        padding_value=PAD_ID,
    )

    return (
        padded_sources.to(DEVICE),
        source_lengths.to(DEVICE),
        [target for _, target in samples],
    )


# ============================================================
# 5. Greedy autoregressive decoding
# ============================================================
def generate_predictions(source, source_lengths):
    batch_size = source.size(0)

    with torch.no_grad():
        # Encode each input only once.
        encoder_state = model.encoder(
            source,
            source_lengths,
        )

        decoder_state = model._bridge_state(encoder_state)

        current_token = torch.full(
            (batch_size, 1),
            SOS_ID,
            dtype=torch.long,
            device=DEVICE,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=DEVICE,
        )

        predictions = [[] for _ in range(batch_size)]

        for _ in range(MAX_OUTPUT_LENGTH):
            logits, decoder_state = model.decoder(
                current_token,
                decoder_state,
            )

            # Choose the highest-probability next character.
            next_token = logits[:, -1, :].argmax(dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue

                token_id = int(next_token[i].item())

                if token_id == EOS_ID:
                    finished[i] = True
                elif token_id not in (PAD_ID, SOS_ID):
                    predictions[i].append(token_id)

            # Finished sequences receive EOS on later steps.
            next_token = torch.where(
                finished,
                torch.full_like(next_token, EOS_ID),
                next_token,
            )

            current_token = next_token.unsqueeze(1)

            if finished.all():
                break

    return predictions


# ============================================================
# 6. Convert IDs to text and calculate edit distance
# ============================================================
def decode_tokens(token_ids):
    return "".join(
        id_to_target[token_id]
        for token_id in token_ids
        if token_id in id_to_target
    )


def clean_reference(target_ids):
    cleaned = []

    for token_id in target_ids:
        token_id = int(token_id)

        if token_id == EOS_ID:
            break

        if token_id not in (PAD_ID, SOS_ID):
            cleaned.append(token_id)

    return cleaned


def levenshtein_distance(reference, prediction):
    """Character-level Levenshtein edit distance."""
    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, pred_char in enumerate(prediction, start=1):
            substitution_cost = int(ref_char != pred_char)

            current.append(
                min(
                    current[-1] + 1,          # insertion
                    previous[j] + 1,           # deletion
                    previous[j - 1]
                    + substitution_cost,       # substitution
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# 7. Evaluate every test example
# ============================================================
total_samples = 0
exact_matches = 0
total_edit_errors = 0
total_reference_characters = 0

sample_predictions = []

with torch.no_grad():
    for start in range(0, len(test_data), BATCH_SIZE):
        batch_samples = test_data[start:start + BATCH_SIZE]

        source, source_lengths, references = prepare_batch(
            batch_samples
        )

        predictions = generate_predictions(
            source,
            source_lengths,
        )

        for predicted_ids, reference_ids in zip(
            predictions, references
        ):
            reference_ids = clean_reference(reference_ids)

            predicted_text = decode_tokens(predicted_ids)
            reference_text = decode_tokens(reference_ids)

            total_samples += 1

            if predicted_text == reference_text:
                exact_matches += 1

            edit_errors = levenshtein_distance(
                reference_text,
                predicted_text,
            )

            total_edit_errors += edit_errors
            total_reference_characters += len(reference_text)

            if len(sample_predictions) < 20:
                sample_predictions.append({
                    "reference": reference_text,
                    "prediction": predicted_text,
                    "exact_match": (
                        predicted_text == reference_text
                    ),
                })

        print(
            f"Evaluated {min(start + BATCH_SIZE, len(test_data))}"
            f" / {len(test_data)} samples"
        )


# ============================================================
# 8. Calculate final metrics
# ============================================================
word_accuracy = (
    100.0 * exact_matches / total_samples
    if total_samples else 0.0
)

if total_reference_characters:
    character_error_rate = (
        total_edit_errors / total_reference_characters
    )
    character_accuracy = (
        100.0 * (1.0 - character_error_rate)
    )
else:
    character_error_rate = 0.0
    character_accuracy = 0.0


# ============================================================
# 9. Display results
# ============================================================
print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print("Total test samples:", total_samples)
print("Exact word matches:", exact_matches)
print(f"Exact-match word accuracy: {word_accuracy:.2f}%")
print(f"Character error rate (CER): {character_error_rate:.4f}")
print(f"Character error rate: {100 * character_error_rate:.2f}%")
print(
    "Character accuracy (100% - CER): "
    f"{character_accuracy:.2f}%"
)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for index, item in enumerate(sample_predictions, start=1):
    print(f"\nExample {index}")
    print("Reference :", item["reference"])
    print("Prediction:", item["prediction"])
    print("Exact match:", item["exact_match"])


# ============================================================
# 10. Save results to JSON
# ============================================================
results = {
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "cell_type": config["cell_type"],
    "embedding_dim": int(config["embedding_dim"]),
    "hidden_dim": int(config["hidden_dim"]),
    "encoder_layers": int(config["encoder_layers"]),
    "decoder_layers": int(config["decoder_layers"]),
    "test_samples": total_samples,
    "exact_word_matches": exact_matches,
    "exact_match_word_accuracy_percent": word_accuracy,
    "character_error_rate": character_error_rate,
    "character_error_rate_percent": (
        100.0 * character_error_rate
    ),
    "character_accuracy_percent": character_accuracy,
    "sample_predictions": sample_predictions,
}

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4,
    )

print("\nResults saved to:")
print(RESULTS_PATH)
