import os
import io
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

root = os.path.join(os.getcwd(), "fitted_results", "pheno")
files = os.listdir(root)
images = [f for f in files if f.endswith(".svg")]

fig, axes = plt.subplots(5, 2, figsize=(10, 10))
axes = axes.flatten()

sorted_images = []
for i in range(1, 11):
    for im in images:
        if im.endswith("s{0}.svg".format(i)):
            sorted_images.append(im)

            # Rasterize the SVG in-memory to PNG bytes, then decode with matplotlib
            png_bytes = cairosvg.svg2png(url=os.path.join(root, im))
            im2plot = mpimg.imread(io.BytesIO(png_bytes))

            ax = axes[i - 1]
            ax.imshow(im2plot)
            ax.set_title("S{0}".format(i), fontsize=12, loc='left')
            ax.axis("off")
            break  # stop searching once we found the match for this index

plt.tight_layout()

fig.subplots_adjust(bottom=0.04, hspace=0, wspace=0)
 
line_handles = [
    Line2D([0], [0], color="black", lw=1.5, label="MEP"),
    Line2D([0], [0], color="red", lw=1.5, label="simMEP"),
]
marker_handles = [
    Line2D([0], [0], marker="o", color="black", linestyle="None",
           markerfacecolor="none", markersize=6, label="MEP"),
    Line2D([0], [0], marker="o", color="red", linestyle="None",
           markerfacecolor="none", markersize=6, label="simMEP"),
]
 
# x-anchors for the bottom-left corner of each column (tune if columns shift)
left_col_x = 0.02
right_col_x = 0.52
 
all_legends = []
for col_x in (left_col_x, right_col_x):
    leg_line = fig.legend(
        handles=line_handles,
        loc="lower left",
        bbox_to_anchor=(col_x, 0.0),
        frameon=True,
        fontsize=8,
    )
    leg_marker = fig.legend(
        handles=marker_handles,
        loc="lower left",
        bbox_to_anchor=(col_x + 0.12, 0.0),
        frameon=True,
        fontsize=8,
    )
    all_legends.extend([leg_line, leg_marker])
 
# fig.legend() calls can otherwise overwrite each other; explicitly keep them all
for leg in all_legends:
    fig.add_artist(leg)
 

plt.savefig(os.path.join(os.getcwd(), "scripts", "figures", "fig_result_pheno.png"), dpi=200)
#plt.show()