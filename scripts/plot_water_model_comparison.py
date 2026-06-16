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


def load_spinodals(path):
    fA_arr, spinodal_arr = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 4:
                _, fA, spinodal_wt_frac, _ = parts
            elif len(parts) == 3:
                fA, spinodal_wt_frac, _ = parts
            else:
                raise ValueError(f"Unexpected row in {path}: {line}")
            fA_arr.append(float(fA))
            spinodal_arr.append(float(spinodal_wt_frac))
    return np.array([fA_arr, spinodal_arr])


def main():
    data_dict = {
        "single": load_spinodals(RPA_RESULTS / "initial_spinodals.txt"),
        "six": load_spinodals(RPA_RESULTS / "initial_spinodals_diblock.txt"),
    }

    plt.scatter(data_dict["single"][0],data_dict["single"][1])
    plt.plot(data_dict["single"][0],data_dict["single"][1], label="Single-water")

    plt.scatter(data_dict["six"][0],data_dict["six"][1])
    plt.plot(data_dict["six"][0],data_dict["six"][1], label="Six-waters")

    plt.xlabel(r"Aromatic composition fraction, $f_A$")
    plt.ylabel(r"Spinodal weight fraction, $w_{poly}^*$")

    plt.title("Water cluster size effect on diblock dilute branch spinodal")
    plt.legend(title=r"$N$ = 20")
    plt.savefig(FIGURE_DIR / "water_cluster_size_effect.png")


if __name__ == "__main__":
    main()
