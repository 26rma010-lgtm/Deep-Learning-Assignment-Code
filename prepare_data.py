
from pathlib import Path
import json
import random
import unicodedata

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "hi.translit.sampled.train.tsv"
DEV_FILE = DATA_DIR / "hi.translit.sampled.dev.tsv"
TEST_FILE = DATA_DIR / "hi.translit.sampled.test.tsv"

SEED = 42

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
]


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

def load_split(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least two columns in {file_path}"
        )

    # Dakshina lexicon convention:
    # column 0 = native-script word
    # column 1 = Romanized word
    df = df.iloc[:, :2].copy()
    df.columns = ["target", "source"]

    # Normalize Unicode consistently.
    df["source"] = df["source"].map(
        lambda x: unicodedata.normalize("NFC", x.strip())
    )
    df["target"] = df["target"].map(
        lambda x: unicodedata.normalize("NFC", x.strip())
    )

    # Remove invalid empty examples.
    df = df[
        (df["source"] != "")
        & (df["target"] != "")
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No valid examples found in {file_path}"
        )

    return df


# --------------------------------------------------
# 3. Build character vocabularies from training only
# --------------------------------------------------

def build_vocab(strings):
    characters = sorted({
        character
        for word in strings
        for character in word
    })

    tokens = SPECIAL_TOKENS + [
        character
        for character in characters
        if character not in SPECIAL_TOKENS
    ]

    token_to_id = {
        token: index
        for index, token in enumerate(tokens)
    }

    return {
        "tokens": tokens,
        "token_to_id": token_to_id,
    }


# --------------------------------------------------
# 4. Encode words as character IDs
# --------------------------------------------------

def encode_word(word, vocab):
    token_to_id = vocab["token_to_id"]

    ids = [
        token_to_id.get(character, token_to_id[UNK_TOKEN])
        for character in word
    ]

    return ids


def encode_split(df, source_vocab, target_vocab):
    source_sequences = []
    target_sequences = []

    for _, row in df.iterrows():
        source_ids = encode_word(
            row["source"],
            source_vocab,
        )

        target_ids = encode_word(
            row["target"],
            target_vocab,
        )

        # The decoder predicts the end token as well.
        source_sequences.append(
            source_ids + [
                source_vocab["token_to_id"][EOS_TOKEN]
            ]
        )

        target_sequences.append(
            [target_vocab["token_to_id"][SOS_TOKEN]]
            + target_ids
            + [target_vocab["token_to_id"][EOS_TOKEN]]
        )

    return {
        "source": source_sequences,
        "target": target_sequences,
    }


# --------------------------------------------------
# 5. Save outputs
# --------------------------------------------------

def save_json(data, file_path):
    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Hindi Dakshina dataset...")

    train_df = load_split(TRAIN_FILE)
    dev_df = load_split(DEV_FILE)
    test_df = load_split(TEST_FILE)

    print(f"Training examples: {len(train_df)}")
    print(f"Development examples: {len(dev_df)}")
    print(f"Test examples: {len(test_df)}")

    # Vocabularies are built from training data only
    # to avoid using development or test data.
    source_vocab = build_vocab(train_df["source"])
    target_vocab = build_vocab(train_df["target"])

    print(
        f"Source vocabulary size: "
        f"{len(source_vocab['tokens'])}"
    )
    print(
        f"Target vocabulary size: "
        f"{len(target_vocab['tokens'])}"
    )

    train_encoded = encode_split(
        train_df,
        source_vocab,
        target_vocab,
    )

    dev_encoded = encode_split(
        dev_df,
        source_vocab,
        target_vocab,
    )

    test_encoded = encode_split(
        test_df,
        source_vocab,
        target_vocab,
    )

    save_json(
        source_vocab,
        OUTPUT_DIR / "source_vocab.json",
    )
    save_json(
        target_vocab,
        OUTPUT_DIR / "target_vocab.json",
    )

    save_json(
        train_encoded,
        OUTPUT_DIR / "train.json",
    )
    save_json(
        dev_encoded,
        OUTPUT_DIR / "dev.json",
    )
    save_json(
        test_encoded,
        OUTPUT_DIR / "test.json",
    )

    # Save clean word pairs for later evaluation.
    for name, df in [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]:
        df.to_csv(
            OUTPUT_DIR / f"{name}_clean.tsv",
            sep="\t",
            index=False,
            encoding="utf-8",
        )

    print("\nExample:")
    print("Romanized input:", train_df.iloc[0]["source"])
    print("Hindi target:", train_df.iloc[0]["target"])
    print(
        "Encoded source:",
        train_encoded["source"][0],
    )
    print(
        "Encoded target:",
        train_encoded["target"][0],
    )

    print("\nPreprocessing completed successfully.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
