
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
BEAM_SIZE = 5
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

    # Each beam: tokens, cumulative log probability,
    # decoder state, next input token, finished flag.
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
                input_token,
                state,
                encoder_outputs,
                source_mask,
            )

            log_probs = torch.log_softmax(
                logits.squeeze(0), dim=-1
            )

            values, indices = torch.topk(
                log_probs,
                k=min(beam_size, log_probs.numel()),
            )

            for log_prob, token_tensor in zip(values, indices):
                token_id = token_tensor.item()

                candidates.append(
                    (
                        tokens + [token_id],
                        score + log_prob.item(),
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

    test_dataset = TransliterationDataset(
        DATA_DIR / "test.json"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    sos_id = target_vocab["token_to_id"][SOS_TOKEN]
    eos_id = target_vocab["token_to_id"][EOS_TOKEN]

    total_words = 0
    exact_matches = 0
    total_edit_distance = 0
    total_reference_characters = 0
    examples = []

    with torch.no_grad():
        for sources, source_lengths, _, labels in test_loader:
            for i in range(sources.size(0)):
                source = sources[i:i + 1].to(device)
                source_length = source_lengths[i:i + 1].to(device)
                label_ids = labels[i].tolist()

                reference = decode_ids(
                    [sos_id] + label_ids,
                    target_vocab,
                )

                prediction_ids = beam_search_decode(
                    model=model,
                    source=source,
                    source_lengths=source_length,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    max_length=len(label_ids) + 10,
                    beam_size=BEAM_SIZE,
                )

                prediction = decode_ids(
                    prediction_ids,
                    target_vocab,
                )

                source_text = decode_ids(
                    source[0].cpu().tolist(),
                    source_vocab,
                )

                exact_matches += int(prediction == reference)
                total_words += 1

                total_edit_distance += levenshtein_distance(
                    reference, prediction
                )
                total_reference_characters += len(reference)

                if len(examples) < 50:
                    examples.append({
                        "source": source_text,
                        "reference": reference,
                        "prediction": prediction,
                    })

            if total_words % 500 < sources.size(0):
                print(f"Evaluated {total_words} test examples")

    word_accuracy = (
        100.0 * exact_matches / max(total_words, 1)
    )

    cer = (
        100.0 * total_edit_distance
        / max(total_reference_characters, 1)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": "test",
        "test_examples": total_words,
        "exact_matches": exact_matches,
        "word_accuracy_percent": word_accuracy,
        "character_error_rate_percent": cer,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_dev_loss": checkpoint["best_dev_loss"],
        "architecture": "GRU with Bahdanau attention",
        "decoding": "beam search",
        "beam_size": BEAM_SIZE,
        "length_penalty_alpha": LENGTH_PENALTY_ALPHA,
    }

    metrics_path = RESULTS_DIR / "attention_beam5_test_metrics.json"
    examples_path = RESULTS_DIR / "attention_beam5_test_examples.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(examples_path, "w", encoding="utf-8") as file:
        json.dump(examples, file, indent=2, ensure_ascii=False)

    print("\nFinal test results")
    print("-" * 45)
    print("Test examples:", total_words)
    print("Exact matches:", exact_matches)
    print(f"Word accuracy: {word_accuracy:.2f}%")
    print(f"Character error rate: {cer:.2f}%")
    print("Beam size:", BEAM_SIZE)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Metrics saved to:", metrics_path)
    print("Examples saved to:", examples_path)

    print("\nSample predictions")
    for example in examples[:10]:
        print(example)


if __name__ == "__main__":
    evaluate()
