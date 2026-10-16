
import json
from pathlib import Path


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

REQUIRED_FILES = [
    "train.json",
    "dev.json",
    "source_vocab.json",
    "target_vocab.json",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def check_dataset_file(filename):
    path = DATA_DIR / filename

    if not path.exists():
        print(f"MISSING: {filename}")
        return None

    try:
        data = load_json(path)
    except (json.JSONDecodeError, OSError) as error:
        print(f"ERROR reading {filename}: {error}")
        return None

    print(f"FOUND: {filename}")

    if isinstance(data, dict):
        print("  Top-level keys:", list(data.keys()))
    elif isinstance(data, list):
        print("  Number of records:", len(data))
    else:
        print("  Data type:", type(data).__name__)

    return data


def inspect_vocab(name, vocab):
    if vocab is None:
        return

    print(f"\n{name} vocabulary:")

    if isinstance(vocab, dict):
        print("  Keys:", list(vocab.keys()))

        tokens = vocab.get("tokens")
        token_to_id = vocab.get("token_to_id")

        if isinstance(tokens, list):
            print("  Vocabulary size:", len(tokens))
            print("  First tokens:", tokens[:10])

        if isinstance(token_to_id, dict):
            print("  Mapping size:", len(token_to_id))
            for token in ["<pad>", "<sos>", "<eos>"]:
                print(f"  {token} ID:", token_to_id.get(token))

        if tokens is None and token_to_id is None:
            print("  Vocabulary structure needs inspection.")

    elif isinstance(vocab, list):
        print("  Vocabulary size:", len(vocab))
        print("  First tokens:", vocab[:10])

    else:
        print("  Unexpected vocabulary format.")


def inspect_split(name, data):
    print(f"\n{name} dataset:")

    if data is None:
        print("  Cannot inspect: file missing or unreadable.")
        return

    if isinstance(data, dict):
        sources = data.get("source")
        targets = data.get("target")

        if sources is None or targets is None:
            print("  Expected keys 'source' and 'target' were not found.")
            print("  Actual keys:", list(data.keys()))
            return

    elif isinstance(data, list):
        if not data:
            print("  Dataset is empty.")
            return

        print("  Records are stored as a list.")
        print("  First record:", repr(data[0])[:300])
        return

    else:
        print("  Unsupported dataset structure.")
        return

    print("  Source examples:", len(sources))
    print("  Target examples:", len(targets))

    if len(sources) != len(targets):
        print("  ERROR: source and target counts do not match!")
        return

    print("  Source-target counts match.")

    if sources:
        print("  First encoded source:", sources[0])
        print("  First encoded target:", targets[0])

        if isinstance(sources[0], list):
            print("  First source length:", len(sources[0]))

        if isinstance(targets[0], list):
            print("  First target length:", len(targets[0]))


def main():
    print("=" * 55)
    print("PROCESSED DATASET CHECK")
    print("=" * 55)
    print("Dataset directory:", DATA_DIR)

    if not DATA_DIR.exists():
        print("\nERROR: The processed data directory does not exist.")
        print("Check the location of your data folder.")
        return

    results = {}

    print("\nChecking required files...")

    for filename in REQUIRED_FILES:
        results[filename] = check_dataset_file(filename)

    print("\n" + "=" * 55)
    print("VOCABULARY CHECK")
    print("=" * 55)

    inspect_vocab(
        "Source",
        results.get("source_vocab.json"),
    )
    inspect_vocab(
        "Target",
        results.get("target_vocab.json"),
    )

    print("\n" + "=" * 55)
    print("DATA SPLIT CHECK")
    print("=" * 55)

    inspect_split("Training", results.get("train.json"))
    inspect_split("Development", results.get("dev.json"))

    print("\n" + "=" * 55)
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (DATA_DIR / filename).exists()
    ]

    if missing:
        print("CHECK RESULT: Some required files are missing.")
        print("Missing files:", missing)
    else:
        print("CHECK RESULT: All four required files exist.")
        print("Review the dataset and vocabulary output above.")

    print("=" * 55)


if __name__ == "__main__":
    main()
