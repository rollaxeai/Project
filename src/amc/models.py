"""Нейросетевые модели классификации вида модуляции.

* :class:`VTCNN2` — эталон из статьи O'Shea, Corgan, Clancy (2016) «Convolutional Radio Modulation
  Recognition Networks»: две свёртки + полносвязный слой; нужен для сравнения с литературой;
* :class:`ResNet1D` — остаточная одномерная свёрточная сеть: устойчиво обучается при большой глубине,
  global average pooling делает её независимой от длины окна (128 / 1024 / 2048);
* :class:`CNNLSTM` — свёрточный «фронт» выделяет локальные формы сигнала, LSTM — их последовательность
  (полезно для ФМн/ЧМн, где информация в смене символов).

У всех моделей есть ``encode(x) → эмбеддинг`` и ``head`` — для многозадачной модели (этап 4а)
и визуализации эмбеддингов (t-SNE/UMAP).
"""

from __future__ import annotations

import torch
from torch import nn


class VTCNN2(nn.Module):
    """VT-CNN2 (O'Shea et al., 2016); вход [B, C, n] трактуется как изображение 1 × C × n."""

    def __init__(self, in_ch: int = 2, n_classes: int = 11, length: int = 128, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)), nn.Conv2d(1, 256, (1, 3)), nn.ReLU(), nn.Dropout(dropout),
            nn.ZeroPad2d((2, 2, 0, 0)), nn.Conv2d(256, 80, (in_ch, 3)), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(80 * (length + 4), 256), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(256, n_classes)
        self.emb_dim = 256

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.fc(self.features(x[:, None]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.encode(x))


class _ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, 5, stride, 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, 5, 1, 2, bias=False), nn.BatchNorm1d(c_out))
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Одномерная ResNet: стем + 4 стадии по 2 блока (каналы widths), шаг 2 между стадиями."""

    def __init__(self, in_ch: int = 2, n_classes: int = 11, widths: tuple[int, ...] = (64, 96, 128, 192),
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, widths[0], 7, 1, 3, bias=False), nn.BatchNorm1d(widths[0]),
                                  nn.ReLU(inplace=True))
        layers, c = [], widths[0]
        for i, w in enumerate(widths):
            layers += [_ResBlock(c, w, 1 if i == 0 else 2), _ResBlock(w, w, 1)]
            c = w
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c, n_classes)
        self.emb_dim = c

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.drop(self.pool(self.stages(self.stem(x))).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.encode(x))


class CNNLSTM(nn.Module):
    """Свёрточный фронт (×4 прореживание) + двухслойная LSTM; эмбеддинг — последнее состояние и среднее."""

    def __init__(self, in_ch: int = 2, n_classes: int = 11, conv: int = 64, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(in_ch, conv, 7, 1, 3), nn.BatchNorm1d(conv), nn.ReLU(inplace=True),
            nn.Conv1d(conv, conv, 5, 2, 2), nn.BatchNorm1d(conv), nn.ReLU(inplace=True),
            nn.Conv1d(conv, conv, 5, 2, 2), nn.BatchNorm1d(conv), nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(conv, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)
        self.emb_dim = 2 * hidden

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        h, _ = self.lstm(self.front(x).transpose(1, 2))
        return self.drop(torch.cat([h[:, -1], h.mean(dim=1)], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.encode(x))


def build(name: str, in_ch: int, n_classes: int, length: int) -> nn.Module:
    """Создать модель по имени: vtcnn2 | resnet | cnnlstm."""
    if name == "vtcnn2":
        return VTCNN2(in_ch, n_classes, length)
    if name == "resnet":
        return ResNet1D(in_ch, n_classes)
    if name == "cnnlstm":
        return CNNLSTM(in_ch, n_classes)
    raise ValueError(name)


def n_params(m: nn.Module) -> int:
    """Число обучаемых параметров."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
