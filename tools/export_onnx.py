"""Экспорт многозадачной модели в ONNX и проверка совпадения ответов с PyTorch.

    python -X utf8 tools/export_onnx.py synth_v1_1024_multitask_s2

Выходы ONNX: ``proba`` [B, 11] (softmax) и ``reg`` [B, 6] (параметры в единицах целей, масштаб уже применён).
"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

from amc.data import ROOT
from amc.multitask import TARGETS, MultiTaskNet


class Export(torch.nn.Module):
    """Обёртка: softmax и обратное масштабирование параметров внутри графа."""

    def __init__(self, net: MultiTaskNet, scale: torch.Tensor):
        super().__init__()
        self.net = net
        self.register_buffer("scale", scale)

    def forward(self, iq: torch.Tensor):  # noqa: D102
        logits, reg = self.net(iq)
        return torch.softmax(logits, dim=1), reg * self.scale


run = sys.argv[1]
ck = torch.load(ROOT / "models" / "deep" / run / "model.pt", map_location="cpu")
net = MultiTaskNet(2, 11)
net.load_state_dict(ck["state"])
net.eval()
m = Export(net, ck["scale"].float()).eval()
out = ROOT / "models" / "onnx"
out.mkdir(parents=True, exist_ok=True)
path = out / "amc_multitask.onnx"
x = torch.randn(8, 2, 1024)
torch.onnx.export(m, (x,), str(path), input_names=["iq"], output_names=["proba", "reg"],
                  dynamic_axes={"iq": {0: "batch"}, "proba": {0: "batch"}, "reg": {0: "batch"}}, opset_version=17,
                  dynamo=False)
(out / "amc_multitask.json").write_text(json.dumps({"run": run, "targets": TARGETS, "window": 1024}), encoding="utf-8")

sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
xb = np.random.default_rng(0).standard_normal((256, 2, 1024)).astype(np.float32)
with torch.no_grad():
    pt_p, pt_r = m(torch.from_numpy(xb))
on_p, on_r = sess.run(None, {"iq": xb})
print("макс. расхождение proba:", float(np.abs(pt_p.numpy() - on_p).max()), "reg:", float(np.abs(pt_r.numpy() - on_r).max()))
print("совпадение классов:", float((pt_p.numpy().argmax(1) == on_p.argmax(1)).mean()))
t0 = time.perf_counter()
for _ in range(5):
    sess.run(None, {"iq": xb})
dt = (time.perf_counter() - t0) / 5
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(5):
        m(torch.from_numpy(xb))
dtp = (time.perf_counter() - t0) / 5
print(f"CPU: ONNX Runtime {256 / dt:.0f} окон/с, PyTorch {256 / dtp:.0f} окон/с; файл {path.stat().st_size / 1e6:.1f} МБ")
