
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from model import Seq2Seq
from train import TransliterationDataset, collate_batch


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_attention_model.pt"
RESULTS_DIR = ROOT / "results"

BATCH_SIZE = 128
BEAM_SIZES = [3, 5, 10]
MAX_EVAL_EXAMPLES = None  # None means evaluate the entire dev set
LENGTH_PENALTY_ALPHA = 0.7

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def decode_ids(ids, vocab):
    tokens = vocab["tokens"]
    result = []

    for token_id in ids:
        token = tokens[token_id]

        if token == EOS_TOKEN:
            break

        if token in {PAD_TOKEN, SOS_TOKEN}:
            continue

        if token == UNK_TOKEN:
            result.append("�")
        else:
            result.append(token)

    return "".join(result)


def levenshtein_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1]


def length_penalized_score(log_probability, length):
    length = max(length, 1)
    penalty = ((5.0 + length) / 6.0) ** LENGTH_PENALTY_ALPHA
    return log_probability / penalty


def prepare_encoder(model, source, source_lengths):
    encoder_outputs, state = model.encoder(
        source, source_lengths
    )
    state = model._bridge_state(state)

    positions = torch.arange(
        source.size(1), device=source.device
    ).unsqueeze(0)

    source_mask = positions < source_lengths.unsqueeze(1)

    return encoder_outputs, state, source_mask


def greedy_decode(
    model, source, source_lengths, sos_id, eos_id, max_length
):
    encoder_outputs, state, source_mask = prepare_encoder(
        model, source, source_lengths
    )

    current_token = torch.tensor(
        [sos_id], dtype=torch.long, device=source.device
    )
    predictions = []

    for _ in range(max_length):
        logits, state, _ = model.decoder(
            current_token, state, encoder_outputs, source_mask
        )

        token_id = logits.argmax(dim=-1).item()
        predictions.append(token_id)

        if token_id == eos_id:
            break

        current_token = torch.tensor(
            [token_id], dtype=torch.long, device=source.device
        )

    if not predictions or predictions[-1] != eos_id:
        predictions.append(eos_id)

    return predictions


def beam_search_decode(
    model,
    source,
    source_lengths,
    sos_id,
    eos_id,
    max_length,
    beam_size,
):
    encoder_outputs, initial_state, source_mask = prepare_encoder(
        model, source, source_lengths
    )

    # Each beam contains:
    # tokens, cumulative log probability, decoder state,
    # next input token, finished flag.
    beams = [([], 0.0, initial_state, sos_id, False)]

    for _ in range(max_length):
        candidates = []

        for tokens, score, state, current_token, finished in beams:
            if finished:
                candidates.append(
                    (tokens, score, state, current_token, True)
                )
                continue

            input_token = torch.tensor(
                [current_token],
                dtype=torch.long,
                device=source.device,
            )

            logits, new_state, _ = model.decoder(
                input_token, state, encoder_outputs, source_mask
            )

            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)

            values, indices = torch.topk(
                log_probs, k=min(beam_size, log_probs.numel())
            )

            for log_prob, token_tensor in zip(values, indices):
                token_id = token_tensor.item()
                new_tokens = tokens + [token_id]
                new_score = score + log_prob.item()

                candidates.append(
                    (
                        new_tokens,
                        new_score,
                        new_state,
                        token_id,
                        token_id == eos_id,
                    )
                )

        if not candidates:
            break

        candidates.sort(
            key=lambda beam: length_penalized_score(
                beam[1], len(beam[0])
            ),
            reverse=True,
        )

        beams = candidates[:beam_size]

        if all(beam[4] for beam in beams):
            break

    best_beam = max(
        beams,
        key=lambda beam: length_penalized_score(
            beam[1], len(beam[0])
        ),
    )

    result = best_beam[0]

    if eos_id not in result:
        result.append(eos_id)

    return result


def update_metrics(metrics, reference, prediction):
    metrics["total"] += 1

    if prediction == reference:
        metrics["exact_matches"] += 1

    metrics["edit_distance"] += levenshtein_distance(
        reference, prediction
    )
    metrics["reference_characters"] += len(reference)


def finalize_metrics(metrics):
    total = metrics["total"]

    return {
        "examples": total,
        "exact_matches": metrics["exact_matches"],
        "word_accuracy_percent": (
            100.0 * metrics["exact_matches"] / max(total, 1)
        ),
        "character_error_rate_percent": (
            100.0 * metrics["edit_distance"]
            / max(metrics["reference_characters"], 1)
        ),
    }


def evaluate():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    source_vocab = load_json(DATA_DIR / "source_vocab.json")
    target_vocab = load_json(DATA_DIR / "target_vocab.json")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint["config"]

    if not config.get("attention", False):
        raise ValueError("The checkpoint is not an attention model.")

    model = Seq2Seq(
        input_vocab_size=len(source_vocab["tokens"]),
        target_vocab_size=len(target_vocab["tokens"]),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        cell_type=config["cell_type"],
        dropout=config["dropout"],
        padding_idx=0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dev_dataset = TransliterationDataset(DATA_DIR / "dev.json")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    sos_id = target_vocab["token_to_id"][SOS_TOKEN]
    eos_id = target_vocab["token_to_id"][EOS_TOKEN]

    methods = ["greedy"] + [f"beam_{size}" for size in BEAM_SIZES]

    metrics = {
        method: {
            "total": 0,
            "exact_matches": 0,
            "edit_distance": 0,
            "reference_characters": 0,
        }
        for method in methods
    }

    examples = []
    processed = 0

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Dataset: development")
    print("Beam sizes:", BEAM_SIZES)
    print("Maximum examples:", MAX_EVAL_EXAMPLES or "all")

    with torch.no_grad():
        for sources, source_lengths, _, labels in dev_loader:
            for i in range(sources.size(0)):
                if (
                    MAX_EVAL_EXAMPLES is not None
                    and processed >= MAX_EVAL_EXAMPLES
                ):
                    break

                source = sources[i:i + 1].to(device)
                source_length = source_lengths[i:i + 1].to(device)
                label_ids = labels[i].tolist()

                reference = decode_ids(
                    [sos_id] + label_ids, target_vocab
                )
                source_text = decode_ids(
                    source[0].cpu().tolist(), source_vocab
                )
                max_length = len(label_ids) + 10

                greedy_ids = greedy_decode(
                    model, source, source_length,
                    sos_id, eos_id, max_length
                )
                greedy_text = decode_ids(greedy_ids, target_vocab)

                update_metrics(
                    metrics["greedy"], reference, greedy_text
                )

                example = {
                    "source": source_text,
                    "reference": reference,
                    "greedy": greedy_text,
                }

                for beam_size in BEAM_SIZES:
                    method = f"beam_{beam_size}"

                    prediction_ids = beam_search_decode(
                        model, source, source_length,
                        sos_id, eos_id, max_length, beam_size
                    )
                    prediction_text = decode_ids(
                        prediction_ids, target_vocab
                    )

                    update_metrics(
                        metrics[method], reference, prediction_text
                    )
                    example[method] = prediction_text

                if len(examples) < 20:
                    examples.append(example)

                processed += 1

            if (
                MAX_EVAL_EXAMPLES is not None
                and processed >= MAX_EVAL_EXAMPLES
            ):
                break

            if processed and processed % 100 == 0:
                print(f"Processed {processed} examples")

    summary = {
        method: finalize_metrics(values)
        for method, values in metrics.items()
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "architecture": "GRU with Bahdanau attention",
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset": "development",
        "examples_evaluated": processed,
        "length_penalty_alpha": LENGTH_PENALTY_ALPHA,
        "results": summary,
    }

    metrics_path = RESULTS_DIR / "dev_decoding_metrics.json"
    examples_path = RESULTS_DIR / "dev_decoding_examples.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    with open(examples_path, "w", encoding="utf-8") as file:
        json.dump(examples, file, indent=2, ensure_ascii=False)

    print("\nDevelopment decoding comparison")
    print("-" * 65)

    for method, result in summary.items():
        print(
            f"{method:10s} | "
            f"Word accuracy: {result['word_accuracy_percent']:.2f}% | "
            f"CER: {result['character_error_rate_percent']:.2f}% | "
            f"Exact matches: {result['exact_matches']}"
        )

    print("\nSample predictions")
    for example in examples[:10]:
        print(example)

    print("\nMetrics saved to:", metrics_path)
    print("Examples saved to:", examples_path)


if __name__ == "__main__":
    evaluate()
