
import torch

from model import Seq2Seq, count_parameters


def test_model(cell_type, encoder_layers, decoder_layers):
    print(
        f"\nTesting {cell_type} | "
        f"Encoder layers: {encoder_layers} | "
        f"Decoder layers: {decoder_layers}"
    )

    batch_size = 4
    source_length = 5
    target_length = 6

    source_vocab_size = 30
    target_vocab_size = 67
    embedding_dim = 32
    hidden_dim = 64
    padding_idx = 0

    model = Seq2Seq(
        input_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        cell_type=cell_type,
        dropout=0.2,
        padding_idx=padding_idx,
    )

    # Synthetic source sequences.
    source = torch.randint(
        1,
        source_vocab_size,
        (batch_size, source_length),
    )

    # Source lengths must be positive and no greater than
    # the padded source sequence length.
    source_lengths = torch.tensor([5, 4, 3, 5])

    # Synthetic decoder inputs.
    target_input = torch.randint(
        1,
        target_vocab_size,
        (batch_size, target_length),
    )

    model.eval()

    with torch.no_grad():
        output = model(
            source,
            target_input,
            source_lengths=source_lengths,
        )

    expected_shape = (
        batch_size,
        target_length,
        target_vocab_size,
    )

    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "Output contains NaN or infinite values."
    )

    parameter_count = count_parameters(model)

    assert parameter_count > 0, (
        "The model has no trainable parameters."
    )

    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", parameter_count)
    print("Test passed.")


def main():
    torch.manual_seed(42)

    cell_types = ["RNN", "GRU", "LSTM"]

    for cell_type in cell_types:
        # Test equal encoder/decoder layer counts.
        test_model(
            cell_type=cell_type,
            encoder_layers=1,
            decoder_layers=1,
        )

        # Test different encoder/decoder layer counts.
        test_model(
            cell_type=cell_type,
            encoder_layers=2,
            decoder_layers=3,
        )

    print("\nAll basic Seq2Seq tests passed successfully.")


if __name__ == "__main__":
    main()
