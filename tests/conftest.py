"""Gera uma mini base sintética no formato PADS para os testes."""
import json

import numpy as np
import pytest


@pytest.fixture
def base_sintetica(tmp_path):
    rng = np.random.default_rng(0)
    raw, pac = tmp_path / "raw", tmp_path / "patients"
    raw.mkdir()
    pac.mkdir()
    condicoes = ["Healthy"] * 8 + ["Parkinson's"] * 8 + ["Other Movement Disorders"] * 2
    t = np.arange(2048) / 100
    for i, c in enumerate(condicoes, 1):
        amp = 0.02 if c == "Healthy" else 0.05
        giro = amp * rng.standard_normal((2048, 3))
        if c != "Healthy":
            giro += 0.03 * np.sin(2 * np.pi * 5 * t)[:, None]
        acc = rng.standard_normal((2048, 3))
        np.savetxt(raw / f"{i:03d}_Relaxed_RightWrist.txt", np.c_[t, acc, giro], delimiter=",")
        meta = dict(id=f"{i:03d}", condition=c, age=int(rng.integers(50, 80)),
                    gender=["male", "female"][i % 2],
                    disease_comment="Atypical Parkinsonism" if i == 12 else "")
        (pac / f"patient_{i:03d}.json").write_text(json.dumps(meta))
    return tmp_path
