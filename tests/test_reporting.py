from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk_lab.reporting import markdown_table, plot_binnings
from credit_risk_lab.woe import WoEEncoder


def test_markdown_table() -> None:
    df = pd.DataFrame(
        {"model": ["Logit", "a|b"], "gini": [0.41234, np.nan], "n": [1200, 3]}
    ).set_index("model")
    lines = markdown_table(df, float_format=".2f").splitlines()
    assert lines[0] == "| model | gini | n |"
    assert lines[1] == "|---|---:|---:|"
    assert lines[2] == "| Logit | 0.41 | 1,200 |"
    assert lines[3] == "| a\\|b |  | 3 |"
    assert markdown_table(df, index=False).splitlines()[0] == "| gini | n |"


def test_plot_binnings(credit_sample: pd.DataFrame) -> None:
    X = credit_sample[["fico", "dti", "purpose", "noise"]]
    encoder = WoEEncoder().fit(X, credit_sample["target"])
    fig = plot_binnings(encoder.binnings_, ncols=3, label_width=6)
    assert sum(ax.get_visible() for ax in fig.axes) == 4
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert all(len(label) <= 6 for label in labels)
    assert any(label.endswith("…") for label in labels)
    plt.close("all")
