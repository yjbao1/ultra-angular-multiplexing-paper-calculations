import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2

import matplotlib.patches as patches
import matplotlib.colors as mcolors

import seaborn as sns
import string
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def make_soft_signal_colormap(name, color_points):
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        color_points,
        N=256,
    )


CMAP_DESIGN = make_soft_signal_colormap(
    "soft_blue_signal",
    [
        (0.00, "#f2fbff"),
        (0.20, "#dceff8"),
        (0.40, "#a8d5ed"),
        (0.62, "#5aa9d5"),
        (0.82, "#1f78b4"),
        (1.00, "#08306b"),
    ],
)

CMAP_SIMULATION = make_soft_signal_colormap(
    "soft_green_signal",
    [
        (0.00, "#f7fff5"),
        (0.20, "#e6f6e2"),
        (0.40, "#bfe5b6"),
        (0.62, "#6fc17a"),
        (0.82, "#2c944c"),
        (1.00, "#005a32"),
    ],
)

CMAP_EXPERIMENT = make_soft_signal_colormap(
    "soft_red_signal",
    [
        (0.00, "#fff2ec"),
        (0.18, "#fee4dc"),
        (0.36, "#fcbdaa"),
        (0.58, "#fb7b66"),
        (0.78, "#dc2f2f"),
        (0.92, "#9f111b"),
        (1.00, "#42000a"),
    ],
)

my_cmap1 = CMAP_EXPERIMENT
CONFUSION_CMAP = "Blues"

THEORY_IMAGE_ROT90_K = 1
EXP_IMAGE_ROT90_K = 0
COLORBAR_WIDTH = 0.008
COLORBAR_PAD = 0.008
RIGHT_PANEL_SHIFT = 0.025
FONT_SCALE = 1.5

fresh_bg_color = (0.94, 0.96, 0.97, 0.37)
# Use white grid lines to contrast with the pale background.
grid_color = 'white'

colors = ['#5DADE2', '#FFB347', '#8FD173', '#DAA520', '#4CAF50']  # Light blue, orange, green, gold, dark green.




def load_exp_bmp_stack(image_dir, expected_count=121):
    image_dir = os.path.abspath(image_dir)
    bmp_files = [
        name for name in os.listdir(image_dir)
        if name.lower().endswith('.bmp') and os.path.splitext(name)[0].isdigit()
    ]
    bmp_files = sorted(bmp_files, key=lambda name: int(os.path.splitext(name)[0]))
    if len(bmp_files) != expected_count:
        raise ValueError(f"Expected {expected_count} BMP files in {image_dir}, found {len(bmp_files)}.")

    images = []
    for name in bmp_files:
        image_path = os.path.join(image_dir, name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = image.astype(np.float32)
        max_value = np.max(image)
        if max_value > 0:
            image = image / max_value
        images.append(image)
    return np.stack(images, axis=0)



data1 = sio.loadmat(os.path.join(BASE_DIR, 'geometry_data_from_python_121_angles_deta_0.6875493541569879_gap_200.0.mat'))
field=data1['output_field']

crosstalk=np.squeeze(data1['crosstalk'])

field_exp = load_exp_bmp_stack(os.path.join(BASE_DIR, 'exp', '121-aligned-exposure1p3'))

data_exp = sio.loadmat(os.path.join(BASE_DIR, 'exp', '121-aligned-exposure1p3', 'confusion_results', 'confusion_results.mat'))
crosstalk_exp = np.squeeze(data_exp['crosstalk'])


def crosstalk_row_average(crosstalk_matrix):
    """Return each channel's mean crosstalk to the other 120 channels, excluding the diagonal."""
    crosstalk_matrix = np.asarray(crosstalk_matrix, dtype=float)
    mask = ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
    return crosstalk_matrix[mask].reshape(crosstalk_matrix.shape[0], crosstalk_matrix.shape[1] - 1).mean(axis=1)


def crosstalk_offdiag_mean(crosstalk_matrix):
    crosstalk_matrix = np.asarray(crosstalk_matrix, dtype=float)
    mask = ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
    return crosstalk_matrix[mask].mean()


confusion_vmax = max(np.nanmax(crosstalk), np.nanmax(crosstalk_exp))


def plot_confusion_matrix(ax, crosstalk_matrix, show_ylabel=True):
    image = ax.imshow(
        crosstalk_matrix,
        aspect='equal',
        cmap=CONFUSION_CMAP,
        vmin=0,
        vmax=confusion_vmax,
    )
    tick_positions = [0, 30, 60, 90, 120]
    tick_labels = ['1', '31', '61', '91', '121']
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('Target index', fontsize=7 * FONT_SCALE)
    if show_ylabel:
        ax.set_ylabel('Measured index', fontsize=7 * FONT_SCALE, labelpad=-1.5)
    ax.tick_params(axis='both', labelsize=6 * FONT_SCALE, length=3)
    return image




left_x = 0.08
right_x = 0.06
gap_x1 = 0.08
gap_x2=0.0025


width_x1=(1-left_x-right_x-gap_x2*20-gap_x1)/22

width_x2=gap_x2*10+width_x1*11
right_panel_x = left_x + 11 * width_x1 + 10 * gap_x2 + gap_x1 + RIGHT_PANEL_SHIFT

fig_width = 1



top_y = 0.01
bottom_y = 0.06

width_y1=width_x1
gap_y1 = gap_x2

gap_y2=0.07

width_confusion = width_x2

fig_length = (top_y + bottom_y + 10*gap_y1+11*width_y1+gap_y2+width_confusion)




gap_y1 = gap_y1 / fig_length

top_y = top_y / fig_length
bottom_y = bottom_y / fig_length
width_y1 = width_y1 / fig_length
gap_y2 = gap_y2 / fig_length
width_confusion = width_confusion / fig_length

fig = plt.figure(figsize=(fig_width * 9, fig_length * 9))


# Set each subplot position manually as [left, bottom, width, height].
positions = []



for i in range(121):

    row_index, col_index = divmod(i, 11)

    x_pos = left_x+col_index*(width_x1+gap_x2)
    y_pos = 1 - top_y - width_y1-row_index*(width_y1+gap_y1)
    width_x = width_x1
    width_y = width_y1



    positions.append([x_pos, y_pos, width_x, width_y])



for i in range(121):

    row_index, col_index = divmod(i, 11)

    ax = fig.add_axes(positions[i])  # Use the custom panel position.
    a = field[i]
    cax = ax.imshow(np.rot90(a, THEORY_IMAGE_ROT90_K),aspect='auto', cmap=CMAP_SIMULATION)
    ax.axis('off')  # Hide axes.

    if col_index == 0:
        tick_val = row_index - 5

        # Draw the numeric row label.
        ax.text(-0.1, 0.5, str(tick_val),
                transform=ax.transAxes,
                va='center', ha='right', fontsize=6 * FONT_SCALE)

        # Draw bottom ticks only on the last row.
    if row_index == 10:
        tick_val = col_index - 5
        # Draw the numeric column label.
        ax.text(0.5, -0.1, str(tick_val),
                transform=ax.transAxes,
                va='top', ha='center', fontsize=6 * FONT_SCALE)

    if row_index == 10 and col_index == 5:
        label_str = r"$\theta_x (\Delta\theta)$"
        # Place the axis label below the numeric tick labels.
        ax.text(0.5, -0.75, label_str,
                transform=ax.transAxes,
                va='top', ha='center', fontsize=8 * FONT_SCALE)

    if row_index == 5 and col_index == 0:
        label_str = r"$\theta_y (\Delta\theta)$"

        ax.text(-1.20, 0.5, label_str,
                transform=ax.transAxes,
                va='center', ha='left', rotation=90, fontsize=8 * FONT_SCALE)



    if i==120 :
        colorbar_height = 11*width_y1+10*gap_y1
        cbar_ax = fig.add_axes( [positions[i][0] +width_x1  + COLORBAR_PAD, positions[i][1], COLORBAR_WIDTH, colorbar_height])
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([cax.norm.vmin, cax.norm.vmax])
        cbar.set_ticklabels(['min', 'max'])
        cbar.ax.tick_params(labelsize=6 * FONT_SCALE, length=0, pad=1.5)







confusion_pos = [left_x, bottom_y, width_x2, width_confusion]
ax = fig.add_axes(confusion_pos)
theory_confusion = plot_confusion_matrix(
    ax,
    crosstalk,
    show_ylabel=True,
)
cbar_ax = fig.add_axes([
    confusion_pos[0] + confusion_pos[2] + COLORBAR_PAD,
    confusion_pos[1],
    COLORBAR_WIDTH,
    confusion_pos[3],
])
cbar = fig.colorbar(theory_confusion, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=6 * FONT_SCALE, length=3)
























# Set each subplot position manually as [left, bottom, width, height].
positions = []



for i in range(121):

    row_index, col_index = divmod(i, 11)

    x_pos = right_panel_x + col_index*(width_x1+gap_x2)
    y_pos = 1 - top_y - width_y1-row_index*(width_y1+gap_y1)
    width_x = width_x1
    width_y = width_y1



    positions.append([x_pos, y_pos, width_x, width_y])



for i in range(121):

    row_index, col_index = divmod(i, 11)

    ax = fig.add_axes(positions[i])  # Use the custom panel position.
    a = field_exp[i]
    cax = ax.imshow(np.rot90(a, EXP_IMAGE_ROT90_K),aspect='auto', cmap=CMAP_EXPERIMENT)
    ax.axis('off')  # Hide axes.

    if col_index == 0:
        tick_val = row_index - 5

        # Draw the numeric row label.
        ax.text(-0.1, 0.5, str(tick_val),
                transform=ax.transAxes,
                va='center', ha='right', fontsize=6 * FONT_SCALE)

        # Draw bottom ticks only on the last row.
    if row_index == 10:
        tick_val = col_index - 5
        # Draw the numeric column label.
        ax.text(0.5, -0.1, str(tick_val),
                transform=ax.transAxes,
                va='top', ha='center', fontsize=6 * FONT_SCALE)

    if row_index == 10 and col_index == 5:
        label_str = r"$\theta_x (\Delta\theta)$"
        # Place the axis label below the numeric tick labels.
        ax.text(0.5, -0.75, label_str,
                transform=ax.transAxes,
                va='top', ha='center', fontsize=8 * FONT_SCALE)

    if row_index == 5 and col_index == 0:
        label_str = r"$\theta_y (\Delta\theta)$"

        ax.text(-1.20, 0.5, label_str,
                transform=ax.transAxes,
                va='center', ha='left', rotation=90, fontsize=8 * FONT_SCALE)



    if i==120 :
        colorbar_height = 11*width_y1+10*gap_y1
        cbar_ax = fig.add_axes( [positions[i][0] +width_x1  + COLORBAR_PAD, positions[i][1], COLORBAR_WIDTH, colorbar_height])
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([cax.norm.vmin, cax.norm.vmax])
        cbar.set_ticklabels(['min', 'max'])
        cbar.ax.tick_params(labelsize=6 * FONT_SCALE, length=0, pad=1.5)







confusion_pos = [right_panel_x, bottom_y, width_x2, width_confusion]
ax = fig.add_axes(confusion_pos)
exp_confusion = plot_confusion_matrix(
    ax,
    crosstalk_exp,
    show_ylabel=False,
)
cbar_ax = fig.add_axes([
    confusion_pos[0] + confusion_pos[2] + COLORBAR_PAD,
    confusion_pos[1],
    COLORBAR_WIDTH,
    confusion_pos[3],
])
cbar = fig.colorbar(exp_confusion, cax=cbar_ax, orientation='vertical')
cbar.ax.tick_params(labelsize=6 * FONT_SCALE, length=3)






plt.savefig(
    os.path.join(BASE_DIR, 'Fig4.svg'),
    format='svg',
    dpi=600,
    bbox_inches='tight',
)  # Change the format to PDF or EPS if needed.
