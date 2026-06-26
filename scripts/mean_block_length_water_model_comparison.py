from matplotlib import font_manager

font_manager.fontManager.addfont("/Users/MichaelMeng1/Documents/Graduate School/Spring26/cmu_serif_roman.ttf")

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RPA_RESULTS = PROJECT_ROOT / "results" / "rpa"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use(PROJECT_ROOT / "styles" / "academic.mplstyle")

SINGLE_WATER_DATA = {
    0.5: {
        "sequences": [
            "GAGAGAGAGAGAGAGAGAGA",
            "GGAGAGAGAGAGAGAGAGAA",
            "GGGAGAGAGAGAGAGAGAAA",
            "GGGGAGAGAGAGAGAGAAAA",
            "GGGGGAGAGAGAGAGAAAAA",
            "GGGGGGAGAGAGAGAAAAAA",
            "GGGGGGGAGAGAGAAAAAAA",
            "GGGGGGGGAGAGAAAAAAAA",
            "GGGGGGGGGAGAAAAAAAAA",
            "GGGGGGGGGGAAAAAAAAAA",
        ],
        "L_T": np.array([1, 1.111111111, 1.25, 1.428571429, 1.666666667, 2, 2.5, 3.333333333, 5, 10]),
        "spinodal": np.array([0.07800628, 0.07299271, 0.04099471, 0.02599754, 0.01899876, 0.01599929, 0.01299928, 0.01199965, 0.01099961, 0.01099961]),
    },
    0.4: {
        "sequences": [
            "GAGGAGAGGAGAGGAGAGGA",
            "GGGAGAGGAGAGGAGAGGAA",
            "GGGGAGGAGAGGAGAGGAAA",
            "GGGGGGAGAGGAGAGGAAAA",
            "GGGGGGGAGGAGAGGAAAAA",
            "GGGGGGGGGAGAGGAAAAAA",
            "GGGGGGGGGGAGGAAAAAAA",
            "GGGGGGGGGGGGAAAAAAAA",
        ],
        "L_T": np.array([1, 1.142857143, 1.333333333, 1.6, 2, 2.666666667, 4, 8]),
        "spinodal": np.array([0.08207929, 0.07404836, 0.06601894, 0.04299266, 0.02999622, 0.02399794, 0.02099787, 0.01899857]),
    },
    0.3: {
        "sequences": [
            "GGAGGGAGGGAGGAGGAGGA",
            "GGGGGAGGGAGGAGGAGGAA",
            "GGGGGGGGAGGAGGAGGAAA",
            "GGGGGGGGGGAGGAGGAAAA",
            "GGGGGGGGGGGGAGGAAAAA",
            "GGGGGGGGGGGGGGAAAAAA",
        ],
        "L_T": np.array([1.0, 1.2, 1.5, 2, 3, 6]),
        "spinodal": np.array([0.14239906,0.08013443, 0.06807813, 0.04798854, 0.03699329, 0.03299443]),
    },
    0.2: {
        "sequences": [
            "GGGGGGGGAGGGGAGGGGAA",
            "GGGGGGGGGGGGAGGGGAAA",
            "GGGGGGGGGGGGGGGAAAAA",
        ],
        "L_T": np.array([1.33333, 2, 4]),
        "spinodal": np.array([0.12240834, 0.09024959, 0.0751712]),
    }
}


def load_six_water_spinodals(fA):
    path = RPA_RESULTS / f"initial_spinodals_fA{fA:.1f}_6water_FIXED.txt"
    spinodals_by_sequence = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            sequence, _, spinodal, _ = line.split()
            spinodals_by_sequence[sequence] = float(spinodal)
    return spinodals_by_sequence


def build_data_dict():
    data_dict = {"single": {}, "six": {}}
    for fA, single_data in SINGLE_WATER_DATA.items():
        if fA == 0.2 or fA == 0.3:
            continue
        data_dict["single"][fA] = np.array([single_data["L_T"], single_data["spinodal"]])

        six_water_spinodals = load_six_water_spinodals(fA)
        data_dict["six"][fA] = np.array([
            single_data["L_T"],
            [six_water_spinodals[sequence] for sequence in single_data["sequences"]],
        ])
    return data_dict


def main():
    data_dict = build_data_dict()
    colors = {0.2: "C0", 0.3: "C1", 0.4: "C2", 0.5: "C3"}

    fig, ax = plt.subplots()
    for fA in sorted(data_dict["single"]):
        x_single, y_single = data_dict["single"][fA]
        x_six, y_six = data_dict["six"][fA]

        ax.plot(x_single, y_single, marker="o", linestyle="-", color=colors[fA], label=rf"Single, $f_A={fA:.1f}$",alpha=0.2)
        ax.plot(x_six, y_six, marker="s", linestyle="--", color=colors[fA], label=rf"Six-water, $f_A={fA:.1f}$")
    ax.set_xlabel(r"Mean hydrophobic block length, $L_T$")
    ax.set_ylabel(r"Spinodal weight fraction, $w_{poly}^*$")
    ax.set_title(r"Water model and hydrophobic block length effect on spinodal")
    ax.legend()

    fig.savefig(FIGURE_DIR / "mean_block_length_water_model_comparison.png")
    return data_dict


if __name__ == "__main__":
    main()
