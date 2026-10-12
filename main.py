
from pathlib import Path
import csv
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
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
TEST_FILE = DATA_DIR / "hi.translit.sampled.test.tsv"

BATCH_SIZE = 32

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


# --------------------------------------------------
# 2. Load the data
# --------------------------------------------------
def load_dataset(file_path):
    pairs = []

    with file_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) >= 2:
                hindi_word = row[0].strip()
                roman_word = row[1].strip()

                if hindi_word and roman_word:
                    pairs.append((roman_word, hindi_word))

    return pairs


train_pairs = load_dataset(TRAIN_FILE)
dev_pairs = load_dataset(DEV_FILE)
test_pairs = load_dataset(TEST_FILE)


# --------------------------------------------------
# 3. Build character vocabularies from training data
# --------------------------------------------------
input_chars = set()
target_chars = set()

for roman_word, hindi_word in train_pairs:
    input_chars.update(roman_word)
    target_chars.update(hindi_word)

input_vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(input_chars)
target_vocab = [
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
] + sorted(target_chars)

input_char_to_id = {
    char: idx for idx, char in enumerate(input_vocab)
}
target_char_to_id = {
    char: idx for idx, char in enumerate(target_vocab)
}

input_pad_id = input_char_to_id[PAD_TOKEN]
target_pad_id = target_char_to_id[PAD_TOKEN]
target_sos_id = target_char_to_id[SOS_TOKEN]
target_eos_id = target_char_to_id[EOS_TOKEN]


# --------------------------------------------------
# 4. Convert words into sequences of integer IDs
# --------------------------------------------------
def encode_input(word):
    return [
        input_char_to_id.get(char, input_char_to_id[UNK_TOKEN])
        for char in word
    ]


def encode_target(word):
    char_ids = [
        target_char_to_id.get(char, target_char_to_id[UNK_TOKEN])
        for char in word
    ]
    return [target_sos_id] + char_ids + [target_eos_id]


# --------------------------------------------------
# 5. PyTorch Dataset
# --------------------------------------------------
class TransliterationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        roman_word, hindi_word = self.pairs[index]

        source_ids = encode_input(roman_word)
        target_ids = encode_target(hindi_word)

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


# --------------------------------------------------
# 6. Pad sequences in each batch
# --------------------------------------------------
def collate_batch(batch):
    sources, targets = zip(*batch)

    source_lengths = torch.tensor(
        [len(sequence) for sequence in sources],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [len(sequence) for sequence in targets],
        dtype=torch.long,
    )

    padded_sources = torch.nn.utils.rnn.pad_sequence(
        sources,
        batch_first=True,
        padding_value=input_pad_id,
    )

    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=target_pad_id,
    )

    return (
        padded_sources,
        source_lengths,
        padded_targets,
        target_lengths,
    )


# --------------------------------------------------
# 7. Create data loaders
# --------------------------------------------------
train_dataset = TransliterationDataset(train_pairs)
dev_dataset = TransliterationDataset(dev_pairs)
test_dataset = TransliterationDataset(test_pairs)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)


# --------------------------------------------------
# 8. Verify one batch
# --------------------------------------------------
print("Datasets and DataLoaders created successfully!")
print("Training examples:", len(train_dataset))
print("Development examples:", len(dev_dataset))
print("Test examples:", len(test_dataset))
print("Input vocabulary size:", len(input_vocab))
print("Target vocabulary size:", len(target_vocab))

source_batch, source_lengths, target_batch, target_lengths = next(
    iter(train_loader)
)

print("\nOne training batch:")
print("Source batch shape:", tuple(source_batch.shape))
print("Source lengths shape:", tuple(source_lengths.shape))
print("Target batch shape:", tuple(target_batch.shape))
print("Target lengths shape:", tuple(target_lengths.shape))

print("\nFirst example in this batch:")
print("Source IDs:", source_batch[0].tolist())
print("Source length:", source_lengths[0].item())
print("Target IDs:", target_batch[0].tolist())
print("Target length:", target_lengths[0].item())

print("\nStep 3 completed!")
