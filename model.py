
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def get_rnn_class(cell_type):
    """Return the selected recurrent neural network type."""
    classes = {
        "RNN": nn.RNN,
        "GRU": nn.GRU,
        "LSTM": nn.LSTM,
    }

    cell_type = cell_type.upper()

    if cell_type not in classes:
        raise ValueError("cell_type must be RNN, GRU, or LSTM")

    return classes[cell_type]


class Encoder(nn.Module):
    """Encoder for character-level sequence-to-sequence learning."""

    def __init__(
        self,
        input_vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        cell_type="GRU",
        dropout=0.0,
        padding_idx=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.upper()

        self.embedding = nn.Embedding(
            input_vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.rnn = get_rnn_class(self.cell_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, source, source_lengths):
        embedded = self.embedding(source)
        embedded = self.embedding_dropout(embedded)

        packed = pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, state = self.rnn(packed)

        return state


class Decoder(nn.Module):
    """Decoder that generates target characters."""

    def __init__(
        self,
        target_vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        cell_type="GRU",
        dropout=0.0,
        padding_idx=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.upper()

        self.embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.rnn = get_rnn_class(self.cell_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            target_vocab_size,
        )

    def forward(self, target_input, initial_state):
        embedded = self.embedding(target_input)
        embedded = self.embedding_dropout(embedded)

        outputs, state = self.rnn(
            embedded,
            initial_state,
        )

        logits = self.output_layer(outputs)

        return logits, state


class Seq2Seq(nn.Module):
    """Basic Seq2Seq encoder-decoder without attention."""

    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        encoder_layers=1,
        decoder_layers=1,
        cell_type="GRU",
        dropout=0.0,
        padding_idx=0,
    ):
        super().__init__()

        cell_type = cell_type.upper()

        if cell_type not in {"RNN", "GRU", "LSTM"}:
            raise ValueError(
                "cell_type must be RNN, GRU, or LSTM"
            )

        self.cell_type = cell_type
        self.padding_idx = padding_idx

        self.encoder = Encoder(
            input_vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            cell_type=cell_type,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.decoder = Decoder(
            target_vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            cell_type=cell_type,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        # Convert encoder states to the decoder's layer dimensions.
        self.hidden_bridge = nn.Linear(
            encoder_layers * hidden_dim,
            decoder_layers * hidden_dim,
        )

        # LSTM requires both hidden and cell states.
        self.cell_bridge = (
            nn.Linear(
                encoder_layers * hidden_dim,
                decoder_layers * hidden_dim,
            )
            if cell_type == "LSTM"
            else None
        )

    def _bridge_tensor(self, tensor, bridge):
        # Input shape: [layers, batch_size, hidden_dim]
        batch_size = tensor.size(1)

        tensor = tensor.transpose(0, 1).contiguous()
        tensor = tensor.reshape(batch_size, -1)

        tensor = torch.tanh(bridge(tensor))

        tensor = tensor.reshape(
            batch_size,
            self.decoder.num_layers,
            self.decoder.hidden_dim,
        )

        return tensor.transpose(0, 1).contiguous()

    def _bridge_state(self, state):
        if self.cell_type == "LSTM":
            hidden, cell = state

            hidden = self._bridge_tensor(
                hidden,
                self.hidden_bridge,
            )

            cell = self._bridge_tensor(
                cell,
                self.cell_bridge,
            )

            return hidden, cell

        return self._bridge_tensor(
            state,
            self.hidden_bridge,
        )

    def forward(
        self,
        source,
        target_input,
        source_lengths=None,
    ):
        if source_lengths is None:
            source_lengths = source.ne(
                self.padding_idx
            ).sum(dim=1)

        encoder_state = self.encoder(
            source,
            source_lengths,
        )

        decoder_initial_state = self._bridge_state(
            encoder_state
        )

        logits, _ = self.decoder(
            target_input,
            decoder_initial_state,
        )

        return logits


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
