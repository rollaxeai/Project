"""Аугментации и входные представления для нейросетей (на GPU, пачками).

Аугментации отражают то, что EDA показал в данных и что встречается в реальном эфире:

* **поворот фазы** — начальная фаза несущей случайна (созвездия «размазаны», 01_eda п. 2);
* **малый сдвиг частоты** — остаточная расстройка приёмника;
* **инверсия спектра** (комплексное сопряжение) — приёмники с инверсией I/Q, нижняя/верхняя боковая.

Все преобразования не меняют класс сигнала.
"""

from __future__ import annotations

import math

import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """[B, 2, n] (I, Q) → комплексный [B, n]."""
    return torch.complex(x[:, 0].float(), x[:, 1].float())


def to_iq(z: torch.Tensor) -> torch.Tensor:
    """Комплексный [B, n] → [B, 2, n]."""
    return torch.stack([z.real, z.imag], dim=1)


def normalize_power(x: torch.Tensor) -> torch.Tensor:
    """Каждый пример [2, n] — к единичной средней мощности (решение EDA, п. 3)."""
    p = x.float().pow(2).sum(dim=1).mean(dim=1).clamp_min(1e-20)
    return x / p.sqrt()[:, None, None]


def augment(x: torch.Tensor, gen: torch.Generator | None = None, cfo_max: float = 0.005,
            p_conj: float = 0.5) -> torch.Tensor:
    """Случайные поворот фазы, сдвиг частоты и инверсия спектра."""
    b, _, n = x.shape
    dev = x.device
    z = to_complex(x)
    theta = torch.rand(b, 1, device=dev, generator=gen) * 2 * math.pi
    f = (torch.rand(b, 1, device=dev, generator=gen) * 2 - 1) * cfo_max
    t = torch.arange(n, device=dev)[None, :]
    z = z * torch.exp(1j * (theta + 2 * math.pi * f * t))
    conj = torch.rand(b, device=dev, generator=gen) < p_conj
    z = torch.where(conj[:, None], z.conj(), z)
    return to_iq(z)


def represent(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Входное представление сети.

    * ``iq`` — I и Q (2 канала);
    * ``ap`` — амплитуда и мгновенная частота (2 канала, инвариантны к фазе);
    * ``iqap`` — все четыре канала.
    """
    if kind == "iq":
        return x
    z = to_complex(x)
    amp = z.abs()
    dphi = torch.angle(z[:, 1:] * z[:, :-1].conj()) / math.pi
    dphi = torch.cat([dphi[:, :1], dphi], dim=1)
    ap = torch.stack([amp, dphi], dim=1)
    if kind == "ap":
        return ap
    if kind == "iqap":
        return torch.cat([x, ap], dim=1)
    raise ValueError(kind)


IN_CHANNELS = {"iq": 2, "ap": 2, "iqap": 4}
