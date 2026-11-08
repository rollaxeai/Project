"""Печать текстовых выходов выполненного ноутбука (без рисунков).

Запуск: python tools/nb_outputs.py notebooks/05_product.ipynb
"""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t and "Figure" not in "".join(t):
            print("".join(t)[:3000])
            print("---")
        if o.get("output_type") == "error":
            print("ERROR", o["ename"], o["evalue"])
