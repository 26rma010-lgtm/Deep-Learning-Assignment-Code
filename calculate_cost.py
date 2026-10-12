
# Theoretical forward-pass computational cost
# MAC = one multiply-accumulate operation

SOURCE_VOCAB_SIZE = 28
TARGET_VOCAB_SIZE = 67

# Representative sequence lengths from your displayed batch.
# Replace these with average source/target lengths for a better estimate.
T_SOURCE = 11
T_TARGET = 11

configs = [
    {
        "cell_type": "RNN",
        "embedding_size": 32,
        "hidden_size": 64,
        "encoder_layers": 1,
        "decoder_layers": 1,
    },
    {
        "cell_type": "GRU",
        "embedding_size": 256,
        "hidden_size": 256,
        "encoder_layers": 3,
        "decoder_layers": 1,
    },
    {
        "cell_type": "LSTM",
        "embedding_size": 16,
        "hidden_size": 32,
        "encoder_layers": 3,
        "decoder_layers": 1,
    },
]


def calculate_cost(config):
    m = config["embedding_size"]
    ke = config["hidden_size"]
    kd = config["hidden_size"]
    Le = config["encoder_layers"]
    Ld = config["decoder_layers"]

    cell = config["cell_type"].upper()
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[cell]

    # Encoder recurrent computation
    encoder_macs = gates * T_SOURCE * (
        m * ke + ke * ke + (Le - 1) * 2 * ke * ke
    )

    # Decoder recurrent computation
    decoder_macs = gates * T_TARGET * (
        m * kd + kd * kd + (Ld - 1) * 2 * kd * kd
    )

    # Linear state bridge computation
    bridge_macs = Le * ke * Ld * kd
    if cell == "LSTM":
        bridge_macs *= 2  # hidden-state and cell-state bridges

    # Output projection: decoder hidden state -> target vocabulary
    output_macs = T_TARGET * kd * TARGET_VOCAB_SIZE

    total_macs = (
        encoder_macs
        + decoder_macs
        + bridge_macs
        + output_macs
    )

    return encoder_macs, decoder_macs, bridge_macs, output_macs, total_macs


for config in configs:
    enc, dec, bridge, output, total = calculate_cost(config)

    print("\nConfiguration:", config)
    print(f"Encoder MACs: {enc:,}")
    print(f"Decoder MACs: {dec:,}")
    print(f"Bridge MACs: {bridge:,}")
    print(f"Output-layer MACs: {output:,}")
    print(f"Total forward-pass MACs: {total:,}")
    print(f"Approximate FLOPs (2 x MACs): {2 * total:,}")
