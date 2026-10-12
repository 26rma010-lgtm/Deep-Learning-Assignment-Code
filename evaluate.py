
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import Seq2Seq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
    / "hi.translit.sampled.test.tsv"
)

MODEL_TYPES = ["gru"]
BEAM_SIZES = [1, 3, 5, 10]

BATCH_SIZE = 64
MAX_DECODE_LENGTH = 50

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_pairs(path):
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        keep_default_na=False
    )

    return [
        (str(row.iloc[1]).strip(), str(row.iloc[0]).strip())
        for _, row in df.iterrows()
        if str(row.iloc[1]).strip()
        and str(row.iloc[0]).strip()
    ]


def encode_source(word, vocab):
    return [
        vocab.get(char, vocab[UNK])
        for char in word
    ] + [vocab[EOS]]


class TestDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        roman, hindi = self.pairs[index]

        return (
            torch.tensor(
                encode_source(roman, self.vocab),
                dtype=torch.long
            ),
            roman,
            hindi
        )


def make_collate(vocab):
    def collate(batch):
        sources, romans, references = zip(*batch)

        lengths = torch.tensor(
            [len(source) for source in sources],
            dtype=torch.long
        )

        max_length = max(len(source) for source in sources)

        padded_sources = [
            torch.cat(
                (
                    source,
                    torch.full(
                        (max_length - len(source),),
                        vocab[PAD],
                        dtype=torch.long
                    )
                )
            )
            for source in sources
        ]

        return (
            torch.stack(padded_sources),
            lengths,
            list(romans),
            list(references)
        )

    return collate


# --------------------------------------------------
# Convert output IDs to Hindi text
# --------------------------------------------------

def decode_ids(ids, vocab):
    reverse_vocab = {
        index: token
        for token, index in vocab.items()
    }

    output = []

    for index in ids:
        token = reverse_vocab.get(int(index), UNK)

        if token == EOS:
            break

        if token not in {PAD, SOS, UNK}:
            output.append(token)

    return "".join(output)


# --------------------------------------------------
# Edit distance
# --------------------------------------------------

def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b)
                )
            )

        previous = current

    return previous[-1]


# --------------------------------------------------
# Greedy decoding (beam size 1 baseline)
# --------------------------------------------------

def predict_greedy(model, source, lengths, target_vocab):
    generated = torch.full(
        (source.size(0), 1),
        target_vocab[SOS],
        dtype=torch.long,
        device=DEVICE
    )

    finished = torch.zeros(
        source.size(0),
        dtype=torch.bool,
        device=DEVICE
    )

    with torch.no_grad():
        for _ in range(MAX_DECODE_LENGTH):
            logits = model(source, generated, lengths)

            next_token = logits[:, -1, :].argmax(dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, target_vocab[EOS]),
                next_token
            )

            generated = torch.cat(
                (generated, next_token.unsqueeze(1)),
                dim=1
            )

            finished |= next_token.eq(target_vocab[EOS])

            if finished.all():
                break

    return generated[:, 1:].cpu().tolist()


# --------------------------------------------------
# Beam-search decoding for one input word
# --------------------------------------------------

def predict_beam_one(
    model,
    source,
    length,
    target_vocab,
    beam_size=3
):
    """
    Decode one source word using beam search.

    Each beam contains:
        (generated_token_ids, cumulative_log_probability)

    source shape: [1, source_length]
    length shape: [1]
    """

    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")

    # Each sequence starts with SOS.
    beams = [([target_vocab[SOS]], 0.0)]

    with torch.no_grad():
        for _ in range(MAX_DECODE_LENGTH):
            candidates = []

            for sequence, score in beams:

                # Keep completed sequences without expanding them.
                if sequence[-1] == target_vocab[EOS]:
                    candidates.append((sequence, score))
                    continue

                decoder_input = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=DEVICE
                )

                logits = model(
                    source,
                    decoder_input,
                    length
                )

                log_probs = F.log_softmax(
                    logits[0, -1, :],
                    dim=-1
                )

                # Avoid generating padding, SOS, or unknown tokens.
                log_probs = log_probs.clone()
                log_probs[target_vocab[PAD]] = float("-inf")
                log_probs[target_vocab[SOS]] = float("-inf")
                log_probs[target_vocab[UNK]] = float("-inf")

                top_k = min(beam_size, log_probs.numel())

                values, indices = torch.topk(
                    log_probs,
                    k=top_k
                )

                for value, index in zip(
                    values.tolist(),
                    indices.tolist()
                ):
                    new_sequence = sequence + [index]
                    new_score = score + value

                    candidates.append(
                        (new_sequence, new_score)
                    )

            if not candidates:
                break

            # Keep the beam_size highest-scoring candidates.
            candidates.sort(
                key=lambda item: item[1],
                reverse=True
            )

            beams = candidates[:beam_size]

            # Stop when every retained beam has reached EOS.
            if all(
                sequence[-1] == target_vocab[EOS]
                for sequence, _ in beams
            ):
                break

    # Prefer a completed sequence if one was generated.
    completed = [
        item for item in beams
        if item[0][-1] == target_vocab[EOS]
    ]

    final_candidates = completed if completed else beams

    # Length normalization helps reduce preference for short outputs.
    best_sequence, _ = max(
        final_candidates,
        key=lambda item: (
            item[1] / max(len(item[0]) - 1, 1)
        )
    )

    # Remove SOS. decode_ids stops at EOS.
    return best_sequence[1:]


# --------------------------------------------------
# Load model checkpoint
# --------------------------------------------------

def load_model(kind):
    checkpoint_path = PROJECT_DIR / f"best_{kind}_model.pt"

    if not checkpoint_path.exists():
        print(
            f"Skipping {kind.upper()}: "
            f"{checkpoint_path.name} not found."
        )
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    source_vocab = checkpoint["source_vocab"]
    target_vocab = checkpoint["target_vocab"]

    model = Seq2Seq(
        len(source_vocab),
        len(target_vocab),
        source_vocab[PAD],
        checkpoint["embedding_size"],
        checkpoint["hidden_size"],
        checkpoint["hidden_size"],
        checkpoint["cell_type"],
        checkpoint["encoder_layers"],
        checkpoint["decoder_layers"],
        dropout=checkpoint.get("dropout", 0.0)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, source_vocab, target_vocab


# --------------------------------------------------
# Evaluate one model and beam size
# --------------------------------------------------

def evaluate(kind, pairs, beam_size=1, loaded_model=None):
    if loaded_model is None:
        loaded_model = load_model(kind)

    if loaded_model is None:
        return None, []

    model, checkpoint, source_vocab, target_vocab = loaded_model

    loader = DataLoader(
        TestDataset(pairs, source_vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate(source_vocab)
    )

    rows = []
    exact_matches = 0
    edit_sum = 0
    reference_chars = 0
    similarity_sum = 0.0

    for source, lengths, romans, references in loader:
        source = source.to(DEVICE)

        if beam_size == 1:
            outputs = predict_greedy(
                model,
                source,
                lengths,
                target_vocab
            )
        else:
            outputs = [
                predict_beam_one(
                    model,
                    source[i:i + 1],
                    lengths[i:i + 1],
                    target_vocab,
                    beam_size=beam_size
                )
                for i in range(source.size(0))
            ]

        for roman, reference, ids in zip(
            romans,
            references,
            outputs
        ):
            prediction = decode_ids(ids, target_vocab)

            distance = edit_distance(reference, prediction)
            exact = prediction == reference

            exact_matches += int(exact)
            edit_sum += distance
            reference_chars += len(reference)

            similarity_sum += max(
                0.0,
                1.0 - distance / max(
                    len(reference),
                    len(prediction),
                    1
                )
            )

            rows.append({
                "model": kind.upper(),
                "beam_size": beam_size,
                "romanized_input": roman,
                "reference_hindi": reference,
                "predicted_hindi": prediction,
                "exact_match": exact,
                "edit_distance": distance
            })

    number_examples = max(len(rows), 1)

    summary = {
        "model": kind.upper(),
        "beam_size": beam_size,
        "test_examples": len(rows),
        "exact_matches": exact_matches,
        "whole_word_accuracy": (
            100 * exact_matches / number_examples
        ),
        "normalized_character_similarity": (
            100 * similarity_sum / number_examples
        ),
        "character_error_rate": (
            100 * edit_sum / max(reference_chars, 1)
        ),
        "trainable_parameters": checkpoint.get(
            "trainable_parameters"
        )
    }

    print(
        f"{kind.upper()} (beam={beam_size}): "
        f"whole-word={summary['whole_word_accuracy']:.2f}%, "
        f"character similarity="
        f"{summary['normalized_character_similarity']:.2f}%, "
        f"CER={summary['character_error_rate']:.2f}%"
    )

    return summary, rows


# --------------------------------------------------
# Main evaluation: compare beam sizes
# --------------------------------------------------

def main():
    print("Device:", DEVICE)

    pairs = load_pairs(TEST_FILE)
    print("Test examples:", len(pairs))

    summaries = []
    all_predictions = []

    for kind in MODEL_TYPES:
        loaded_model = load_model(kind)

        if loaded_model is None:
            continue

        for beam_size in BEAM_SIZES:
            summary, rows = evaluate(
                kind,
                pairs,
                beam_size=beam_size,
                loaded_model=loaded_model
            )

            if summary is not None:
                summaries.append(summary)
                all_predictions.extend(rows)

    if summaries:
        summary_path = PROJECT_DIR / "beam_search_results.csv"
        predictions_path = (
            PROJECT_DIR / "beam_search_predictions.csv"
        )

        summary_df = pd.DataFrame(summaries)

        summary_df.to_csv(summary_path, index=False)

        pd.DataFrame(all_predictions).to_csv(
            predictions_path,
            index=False
        )

        print("\nBeam-search comparison:")
        print(summary_df.to_string(index=False))

        print("\nSummary saved:", summary_path)
        print("Predictions saved:", predictions_path)

    else:
        print("No model checkpoints found. Run train.py first.")


if __name__ == "__main__":
    main()
