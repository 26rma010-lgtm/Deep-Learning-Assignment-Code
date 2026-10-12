
from model import Seq2Seq

# Vocabulary sizes from your main.py output
INPUT_VOCAB_SIZE = 28
TARGET_VOCAB_SIZE = 67

# Padding ID shown in your batch output
INPUT_PAD_ID = 0

configs = [
    {
        "cell_type": "RNN",
        "embedding_size": 32,
        "hidden_size": 64,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "dropout": 0.2,
    },
    {
        "cell_type": "GRU",
        "embedding_size": 256,
        "hidden_size": 256,
        "encoder_layers": 3,
        "decoder_layers": 1,
        "dropout": 0.3,
    },
    {
        "cell_type": "LSTM",
        "embedding_size": 16,
        "hidden_size": 32,
        "encoder_layers": 3,
        "decoder_layers": 1,
        "dropout": 0.3,
    },
]

for config in configs:
    model = Seq2Seq(
        input_vocab_size=INPUT_VOCAB_SIZE,
        target_vocab_size=TARGET_VOCAB_SIZE,
        input_pad_id=INPUT_PAD_ID,
        embedding_size=config["embedding_size"],
        encoder_hidden_size=config["hidden_size"],
        decoder_hidden_size=config["hidden_size"],
        cell_type=config["cell_type"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        dropout=config["dropout"],
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nConfiguration:", config)
    print("Source vocabulary:", INPUT_VOCAB_SIZE)
    print("Target vocabulary:", TARGET_VOCAB_SIZE)
    print("Trainable parameters:", total_params)
