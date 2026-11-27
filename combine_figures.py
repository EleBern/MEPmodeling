"""
Combine per-subject figures into summary plots.

Expected layout:
    <base_dir>/S1/E1.png, E2.png, I1.png, I2.png
    <base_dir>/S2/...
    ...
    <base_dir>/S10/...

Creates 4 figures (E1, E2, I1, I2), each a 5x2 grid with one subplot per subject.

Usage:
    python combine_figures.py                 # base_dir = current directory
    python combine_figures.py /path/to/results
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

N_SUBJECTS = 10
N_ROWS, N_COLS = 5, 2
FIGURE_NAMES = ["E1", "E2", "I1", "I2"]


def make_summary_plot(base_dir, fig_name, out_dir):
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(10, 20))
    axes = axes.flatten()

    for i in range(N_SUBJECTS):
        subject = f"S{i + 1}"
        ax = axes[i]
        img_path = os.path.join(base_dir, subject, f"{fig_name}.png")

        if os.path.isfile(img_path):
            ax.imshow(mpimg.imread(img_path))
        else:
            print(f"Warning: missing {img_path}")
            ax.text(0.5, 0.5, "File not found", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_title(subject, fontsize=14)
        ax.axis("off")

    fig.suptitle(fig_name, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(out_dir, f"{fig_name}_all_subjects.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("/home/bernasconi/Documents/Programma/fparam_distribution", "NoNoise", "SyntheticMUAPs")
    out_dir = os.path.join(base_dir, "combined_figures")
    os.makedirs(out_dir, exist_ok=True)

    for fig_name in FIGURE_NAMES:
        make_summary_plot(base_dir, fig_name, out_dir)


if __name__ == "__main__":
    main()