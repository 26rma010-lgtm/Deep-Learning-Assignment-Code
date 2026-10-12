
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Seq2Seq(nn.Module):
    """Character-level Romanized Hindi to Hindi encoder-decoder."""

    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,
        input_pad_id,
        embedding_size=64,
        encoder_hidden_size=128,
        decoder_hidden_size=128,
        cell_type="GRU",
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.cell_type = cell_type.upper()
        self.input_pad_id = input_pad_id
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        if self.cell_type not in {"RNN", "GRU", "LSTM"}:
            raise ValueError("cell_type must be RNN, GRU, or LSTM")

        if encoder_layers < 1 or decoder_layers < 1:
            raise ValueError("Layer counts must be >= 1")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        # Character embeddings
        self.encoder_embedding = nn.Embedding(
            input_vocab_size,
            embedding_size,
            padding_idx=input_pad_id,
        )

        self.decoder_embedding = nn.Embedding(
            target_vocab_size,
            embedding_size,
            padding_idx=0,
        )

        # Dropout after embeddings
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(dropout)

        # Select recurrent cell type
        cell = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }[self.cell_type]

        # Encoder
        self.encoder = cell(
            input_size=embedding_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
        )

        # Decoder
        self.decoder = cell(
            input_size=embedding_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
        )

        # Connect encoder states to decoder states
        bridge_in = encoder_layers * encoder_hidden_size
        bridge_out = decoder_layers * decoder_hidden_size

        self.hidden_bridge = nn.Linear(bridge_in, bridge_out)

        self.cell_bridge = (
            nn.Linear(bridge_in, bridge_out)
            if self.cell_type == "LSTM"
            else None
        )

        # Predict the next Hindi character
        self.output_layer = nn.Linear(
            decoder_hidden_size,
            target_vocab_size,
        )

    def _bridge(self, state, layer):
        # Shape:
        # (encoder_layers, batch, encoder_hidden_size)
        # -> (decoder_layers, batch, decoder_hidden_size)

        batch = state.size(1)

        state = state.transpose(0, 1).contiguous().view(batch, -1)
        state = torch.tanh(layer(state))

        state = state.view(
            batch,
            self.decoder_layers,
            self.decoder_hidden_size,
        )

        return state.transpose(0, 1).contiguous()

    def forward(self, source_ids, target_input_ids, source_lengths=None):
        # Calculate source lengths if not provided
        if source_lengths is None:
            source_lengths = source_ids.ne(self.input_pad_id).sum(1)

        lengths = source_lengths.detach().to("cpu").clamp(min=1)

        # Encode the Romanized input
        embedded = self.encoder_embedding(source_ids)
        embedded = self.encoder_dropout(embedded)

        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, state = self.encoder(packed)

        # Prepare initial decoder state
        if self.cell_type == "LSTM":
            h, c = state

            initial = (
                self._bridge(h, self.hidden_bridge),
                self._bridge(c, self.cell_bridge),
            )
        else:
            initial = self._bridge(state, self.hidden_bridge)

        # Decode the Hindi character sequence
        decoder_embedded = self.decoder_embedding(target_input_ids)
        decoder_embedded = self.decoder_dropout(decoder_embedded)

        decoded, _ = self.decoder(decoder_embedded, initial)

        # Return next-character logits
        return self.output_layer(decoded)
