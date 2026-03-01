"""
model.py
--------
CNN-LSTM hybrid for 1D physiological signal classification (apnea vs normal).

Architecture
~~~~~~~~~~~~
Input  : (B, C, 3000)  C = 1 or 2 signal channels
Output : (B,)          raw logits (apply sigmoid for probability)

Encoder (Conv1D):
  InputProjection : Conv1d(C→2, k=1)         — unify channel count
  Block 1         : Conv1d(2→32,  k=15, p=7) → BN → GELU → MaxPool1d(4)  → (B,32, 750)
  Block 2         : Conv1d(32→64, k=9,  p=4) → BN → GELU → MaxPool1d(4)  → (B,64, 187)
  Block 3         : Conv1d(64→128,k=5,  p=2) → BN → GELU → MaxPool1d(3)  → (B,128, 62)

Temporal (BiLSTM):
  Permute         : (B, 62, 128)
  BiLSTM          : input=128, hidden=64, layers=2, dropout=0.3 → (B,62,128)
  Global avg pool : (B, 128)

Classifier:
  Linear(128→64) → GELU → Dropout(0.4)
  Linear(64→1)                              — logit output
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ApneaCNN_LSTM(nn.Module):
    """
    CNN-LSTM hybrid apnea classifier.

    Parameters
    ----------
    in_channels : int
        Number of input signal channels (1 or 2).
    lstm_hidden : int
        Hidden size per LSTM direction.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout applied inside LSTM and classifier.
    """

    def __init__(
        self,
        in_channels: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project variable-channel input to fixed 2 channels
        self.input_proj = nn.Conv1d(in_channels, 2, kernel_size=1, bias=False)

        # ----------------------------------------------------------------
        # Temporal downsampling: 3000 samples (300 Hz) → 300 samples (30 Hz)
        # This gives kernels a 10× larger effective receptive field so that
        # a kernel_size=61 sees 61/30 = 2 seconds of signal — enough to
        # detect peaks in a 4-second (15 bpm) breathing cycle.
        # ----------------------------------------------------------------
        self.downsample = nn.AvgPool1d(kernel_size=10)

        # CNN encoder operating at 30 Hz effective sample rate
        # ConvBlock(in, out, kernel, pool):
        #   k=61 @ 30 Hz → 2.0 s receptive field  300→100
        #   k=21 @ 10 Hz → 2.1 s receptive field  100→25
        #   k=11 @ 2.5Hz → 4.4 s receptive field   25→5
        self.encoder = nn.Sequential(
            ConvBlock(2,   32,  kernel=61, pool=3),
            ConvBlock(32,  64,  kernel=21, pool=4),
            ConvBlock(64,  128, kernel=11, pool=5),
        )

        # Bidirectional LSTM over 5 temporal steps
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_size = lstm_hidden * 2   # bidirectional

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T)  raw pre-processed signal windows
        Returns:
            logits : (B,)   raw scores (not sigmoid-ed)
        """
        # Input projection: (B, C, T) → (B, 2, T)
        x = self.input_proj(x)

        # Temporal downsampling: (B, 2, 3000) → (B, 2, 300)
        x = self.downsample(x)

        # CNN encoding: (B, 2, 300) → (B, 128, 5)
        x = self.encoder(x)

        # Reshape for LSTM: (B, 128, L) → (B, L, 128)
        x = x.permute(0, 2, 1).contiguous()

        # BiLSTM: (B, L, 128) → (B, L, 128)
        x, _ = self.lstm(x)

        # Global average pooling over time: (B, L, 128) → (B, 128)
        x = x.mean(dim=1)

        # Classifier: (B, 128) → (B, 1) → (B,)
        logits = self.classifier(x).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions (0/1)."""
        return (self.predict_proba(x) >= threshold).long()


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------

def model_summary(model: ApneaCNN_LSTM, in_channels: int = 2, seq_len: int = 3000):
    """Print parameter count and a quick shape trace."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ApneaCNN_LSTM")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Input shape      : (B, {in_channels}, {seq_len})")

    device = next(model.parameters()).device
    dummy = torch.zeros(2, in_channels, seq_len, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Output shape     : {tuple(out.shape)}")


if __name__ == "__main__":
    m = ApneaCNN_LSTM(in_channels=2)
    model_summary(m)
