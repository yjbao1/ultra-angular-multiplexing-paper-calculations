from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


BASE_DIR = Path(__file__).resolve().parent

SWEEP_FILE = "simulation_sweep_results_N_2_case_paper.mat"
N_THETA_FILE = "simulation_N_theta_paper.mat"
DISTANCE_FILE = "simulation_optimized_distance1_paper.mat"
NA_FILE = "NA_scan_crosstalk.mat"


# ---------- Style ----------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "path"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": "#263238",
        "axes.labelcolor": "#263238",
        "xtick.color": "#263238",
        "ytick.color": "#263238",
        "text.color": "#263238",
    }
)

ink = "#263238"
graphite = "#4F5B62"
mist = "#E6E1DA"
guide = "#D6D0C8"
paper = "#FBFAF7"
plot_bg = "#FCFBF8"
panel_bg = "#F7F8F4"
image_plate_bg = "#F2F8F3"
matrix_plate_bg = "#F1F5FA"
crosstalk_color = "#C65A4A"
crosstalk_dark = "#934035"
target_energy_color = "#2A6FB8"
data_color = "#20B8AA"
fit_color = "#111111"
distance_color = "#00A6D6"
na_color = "#F28E2B"
target_color = "#4C9A62"
highlight_gold = "#C9972B"
main_line_width = 1.35
fit_line_width = 1.15

field_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "soft_green_signal",
    [
        (0.00, "#f7fff5"),
        (0.20, "#e6f6e2"),
        (0.40, "#bfe5b6"),
        (0.62, "#6fc17a"),
        (0.82, "#2c944c"),
        (1.00, "#005a32"),
    ],
    N=256,
)
matrix_cmap = plt.get_cmap("Blues")


# ---------- Data: keep the same data sources as the previous Fig2_8 ----------

sweep_data = sio.loadmat(BASE_DIR / SWEEP_FILE, squeeze_me=True, struct_as_record=False)
angle_mrad = np.asarray(sweep_data["deta_angle_mrad_list"], dtype=float).ravel()
crosstalk_average = np.asarray(sweep_data["crosstalk_average_list"], dtype=float).ravel()
confusion_cell = np.asarray(sweep_data["crosstalk_confusing_cell"], dtype=float)
field_cell = np.asarray(sweep_data["output_field1_cell"], dtype=float)
mean_target_energy_fraction = np.mean(
    np.diagonal(confusion_cell, axis1=1, axis2=2),
    axis=1,
)

n_theta_data = sio.loadmat(BASE_DIR / N_THETA_FILE, squeeze_me=True, struct_as_record=False)
n_theta = np.asarray(n_theta_data["N_theta"], dtype=float)
n_list = np.asarray(n_theta[0, :], dtype=float).ravel()
theta_mrad = np.asarray(n_theta[2, :], dtype=float).ravel()
theta_slope, theta_intercept = np.polyfit(n_list, theta_mrad, 1)
theta_fit = theta_slope * n_list + theta_intercept
theta_ss_res = float(np.sum((theta_mrad - theta_fit) ** 2))
theta_ss_tot = float(np.sum((theta_mrad - np.mean(theta_mrad)) ** 2))
theta_r2 = 1.0 - theta_ss_res / theta_ss_tot if theta_ss_tot else np.nan

distance_data = sio.loadmat(BASE_DIR / DISTANCE_FILE, squeeze_me=True, struct_as_record=False)
distance_um_all = np.asarray(distance_data["distance1"], dtype=float).ravel() * 1e6
distance_ct_all = np.asarray(distance_data["crosstalk_average_list"], dtype=float).ravel()
distance_keep = (distance_um_all >= 200 - 1e-9) & (distance_um_all <= 500 + 1e-9)
distance_um = distance_um_all[distance_keep]
distance_ct = distance_ct_all[distance_keep]

na_data = sio.loadmat(BASE_DIR / NA_FILE, squeeze_me=True, struct_as_record=False)
na_list = np.asarray(na_data["NA_list"], dtype=float).ravel()
na_ct = np.asarray(na_data["crosstalk_average"], dtype=float).ravel()


# ---------- Manual layout ----------

fig_width = 7.2
layout_fig_height = 5.55
fig_height = 4.85
bottom_crop_in = layout_fig_height - fig_height


def crop_y(y_value):
    return (y_value * layout_fig_height - bottom_crop_in) / fig_height


def crop_h(height_value):
    return height_value * layout_fig_height / fig_height

# Main panels.
main_panel_width = 0.365
lower_panel_height = crop_h(0.320 * 5.0 / 6.0)

panel_a_left = 0.075
panel_a_bottom = crop_y(0.595)
panel_a_width = main_panel_width
panel_a_height = crop_h(0.315)
panel_a_top = panel_a_bottom + panel_a_height

main_column_gap = 0.090
main_row_gap = crop_h(0.125)

panel_b_left = panel_a_left + panel_a_width + main_column_gap
panel_b_top = panel_a_top

panel_c_left = 0.075
panel_c_width = main_panel_width
panel_c_height = lower_panel_height
panel_c_bottom = panel_a_bottom - main_row_gap - panel_c_height

panel_d_left = panel_b_left
panel_d_bottom = panel_c_bottom
panel_d_width = main_panel_width
panel_d_height = lower_panel_height

# Panel b internal gaps, written like Fig2_7 for easy manual adjustment.
panel_b_tile_width = 0.080
panel_b_tile_height = panel_b_tile_width * fig_width / fig_height
panel_b_column_gap = 0.008
panel_b_row_gap = crop_h(0.009)
panel_b_matrix_gap = crop_h(0.009)
panel_b_cbar_gap = 0.012
panel_b_cbar_width = 0.008

panel_b_row1_bottom = panel_b_top - panel_b_tile_height
panel_b_row2_bottom = panel_b_row1_bottom - panel_b_row_gap - panel_b_tile_height
panel_b_matrix_bottom = panel_b_row2_bottom - panel_b_matrix_gap - panel_b_tile_height
panel_b_column_x = [
    panel_b_left + i * (panel_b_tile_width + panel_b_column_gap)
    for i in range(4)
]
panel_b_right = panel_b_column_x[-1] + panel_b_tile_width
panel_b_center = 0.5 * (panel_b_left + panel_b_right)
panel_b_cbar_left = panel_b_right + panel_b_cbar_gap

fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
fig.patch.set_facecolor("white")

reference_angle_mrad = 0.57
b_panel_angles_mrad = np.array([0.10, 0.30, reference_angle_mrad, 0.90])
b_indices = np.array(
    [int(np.argmin(np.abs(angle_mrad - target_angle))) for target_angle in b_panel_angles_mrad],
    dtype=int,
)
reference_col = int(np.argmin(np.abs(angle_mrad[b_indices] - reference_angle_mrad)))
reference_index = int(b_indices[reference_col])


# ---------- Panel a: angle sweep ----------

ax_a = fig.add_axes([panel_a_left, panel_a_bottom, panel_a_width, panel_a_height])

angle_pad = (float(np.nanmax(angle_mrad)) - float(np.nanmin(angle_mrad))) * 0.03
ct_pad = (float(np.nanmax(crosstalk_average)) - float(np.nanmin(crosstalk_average))) * 0.18
target_energy_pad = (
    float(np.nanmax(mean_target_energy_fraction))
    - float(np.nanmin(mean_target_energy_fraction))
) * 0.18

ax_a.plot(
    angle_mrad,
    crosstalk_average,
    color=crosstalk_color,
    lw=main_line_width,
    marker="p",
    ms=6.3,
    mfc=crosstalk_color,
    mec=crosstalk_color,
    mew=0,
    label="Mean crosstalk",
)
ax_a.set_xlabel(r"$\Delta\theta$ (mrad)")
ax_a.set_ylabel("Mean crosstalk", color=crosstalk_color)
ax_a.set_xlim(float(np.nanmin(angle_mrad)) - angle_pad, float(np.nanmax(angle_mrad)) + angle_pad)
left_y_min = -0.01
left_y_max = 0.21
ax_a.set_ylim(left_y_min, left_y_max)
target_crosstalk = 0.04
left_y_ticks = np.arange(0.0, 0.201, 0.04)
right_y_ticks = 0.75 + left_y_ticks * 1.25
ax_a.set_yticks(left_y_ticks)
ax_a.tick_params(axis="y", colors=crosstalk_color)
ax_a.spines["left"].set_color(crosstalk_color)
ax_a.spines["left"].set_linewidth(0.75)
ax_a.spines["top"].set_visible(True)
ax_a.spines["top"].set_color("black")
ax_a.spines["top"].set_linewidth(0.75)
ax_a.grid(axis="y", lw=0.62, color=mist, alpha=0.80, linestyle=(0, (4.2, 2.0)))

ax_a_right = ax_a.twinx()
ax_a_right.plot(
    angle_mrad,
    mean_target_energy_fraction,
    color=target_energy_color,
    lw=main_line_width,
    marker="v",
    ms=6.3,
    mfc=target_energy_color,
    mec=target_energy_color,
    mew=0,
    label="Mean target energy fraction",
)
ax_a_right.set_ylabel("Mean target energy fraction", color=target_energy_color)
ax_a_right.set_ylim(0.75 + left_y_min * 1.25, 0.75 + left_y_max * 1.25)
ax_a_right.set_yticks(right_y_ticks)
ax_a_right.tick_params(axis="y", colors=target_energy_color)
ax_a_right.spines["right"].set_visible(True)
ax_a_right.spines["right"].set_linewidth(0.75)
ax_a_right.spines["right"].set_color(target_energy_color)
ax_a_right.spines["top"].set_visible(True)
ax_a_right.spines["top"].set_color("black")
ax_a_right.spines["top"].set_linewidth(0.75)
ax_a_right.spines["left"].set_visible(False)
ax_a_right.yaxis.set_ticks_position("right")

ax_a.axvline(reference_angle_mrad, color=highlight_gold, lw=main_line_width, ls=(0, (4.2, 2.0)), zorder=1)


# ---------- Panel b: fields and confusion matrices ----------

b_fields = field_cell[b_indices, :, :, :]
b_matrices = confusion_cell[b_indices, :, :]
b_offdiag_mean = 0.5 * (b_matrices[:, 0, 1] + b_matrices[:, 1, 0])
b_offdiag_span = float(np.nanmax(b_offdiag_mean) - np.nanmin(b_offdiag_mean))
b_vmax = float(np.nanpercentile(b_fields, 99.6))
if b_vmax <= 0:
    b_vmax = float(np.nanmax(b_fields)) or 1.0
b_field_norm = mpl.colors.PowerNorm(gamma=0.72, vmin=0.0, vmax=b_vmax)
b_matrix_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

for col in range(4):
    x_pos = panel_b_column_x[col]
    angle_index = b_indices[col]
    matrix = confusion_cell[angle_index, :, :]
    off_diag_mean = b_offdiag_mean[col]
    is_reference_col = col == reference_col

    fig.text(
        x_pos + panel_b_tile_width / 2,
        panel_b_top + 0.016,
        f"{angle_mrad[angle_index]:.2g} mrad",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )

    # Row 1: calculated image for input A.
    ax_img_a = fig.add_axes([x_pos, panel_b_row1_bottom, panel_b_tile_width, panel_b_tile_height])
    image_a = np.fliplr(np.rot90(b_fields[col, 0, :, :], k=-1))
    ax_img_a.imshow(image_a, cmap=field_cmap, norm=b_field_norm, origin="lower", interpolation="lanczos")
    ax_img_a.set_box_aspect(1)
    ax_img_a.set_xticks([])
    ax_img_a.set_yticks([])
    frame_color = highlight_gold if is_reference_col else "black"
    frame_width = 1.15 if is_reference_col else 0.85
    for spine in ax_img_a.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(frame_width)
        spine.set_zorder(20)
    ax_img_a.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_img_a.transAxes,
            fill=False,
            edgecolor=frame_color,
            linewidth=frame_width,
            zorder=30,
            clip_on=False,
        )
    )

    # Row 2: calculated image for input B.
    ax_img_b = fig.add_axes([x_pos, panel_b_row2_bottom, panel_b_tile_width, panel_b_tile_height])
    image_b = np.fliplr(np.rot90(b_fields[col, 1, :, :], k=-1))
    ax_img_b.imshow(image_b, cmap=field_cmap, norm=b_field_norm, origin="lower", interpolation="lanczos")
    ax_img_b.set_box_aspect(1)
    ax_img_b.set_xticks([])
    ax_img_b.set_yticks([])
    for spine in ax_img_b.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(frame_width)
        spine.set_zorder(20)
    ax_img_b.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_img_b.transAxes,
            fill=False,
            edgecolor=frame_color,
            linewidth=frame_width,
            zorder=30,
            clip_on=False,
        )
    )

    # Row 3: 2 x 2 crosstalk/confusion matrix.
    ax_matrix = fig.add_axes([x_pos, panel_b_matrix_bottom, panel_b_tile_width, panel_b_tile_height])
    ax_matrix.imshow(matrix, cmap=matrix_cmap, norm=b_matrix_norm, interpolation="nearest")
    ax_matrix.set_box_aspect(1)
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_xticklabels(["A", "B"])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_yticklabels(["A", "B"] if col == 0 else [], rotation=0)
    ax_matrix.tick_params(axis="both", which="major", length=0, labelsize=6.8, pad=5.0)
    ax_matrix.set_xticks(np.arange(-0.5, 2.0, 1.0), minor=True)
    ax_matrix.set_yticks(np.arange(-0.5, 2.0, 1.0), minor=True)
    ax_matrix.tick_params(which="minor", bottom=False, left=False)
    for tick_x in [0, 1]:
        ax_matrix.plot(
            [tick_x, tick_x],
            [0, -0.052],
            transform=ax_matrix.get_xaxis_transform(),
            color="black",
            lw=0.65,
            clip_on=False,
            zorder=25,
        )
    for tick_y in [0, 1]:
        ax_matrix.plot(
            [-0.052, 0],
            [tick_y, tick_y],
            transform=ax_matrix.get_yaxis_transform(),
            color="black",
            lw=0.65,
            clip_on=False,
            zorder=25,
        )

    for row in range(2):
        for matrix_col in range(2):
            value = float(matrix[row, matrix_col])
            rgba = matrix_cmap(b_matrix_norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.54 else ink
            ax_matrix.text(
                matrix_col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color=text_color,
            )
    for spine in ax_matrix.spines.values():
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(frame_width)
        spine.set_zorder(20)
    ax_matrix.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_matrix.transAxes,
            fill=False,
            edgecolor=frame_color,
            linewidth=frame_width,
            zorder=30,
            clip_on=False,
        )
    )

field_cbar_bottom = panel_b_row2_bottom
field_cbar_height = 2 * panel_b_tile_height + panel_b_row_gap
matrix_cbar_bottom = panel_b_matrix_bottom
matrix_cbar_height = panel_b_tile_height

field_cbar_ax = fig.add_axes(
    [
        panel_b_cbar_left,
        field_cbar_bottom,
        panel_b_cbar_width,
        field_cbar_height,
    ]
)
field_sm = mpl.cm.ScalarMappable(norm=b_field_norm, cmap=field_cmap)
field_cbar = fig.colorbar(field_sm, cax=field_cbar_ax)
field_cbar.set_ticks([0, b_vmax])
field_cbar.set_ticklabels(["min", "max"])
field_cbar.ax.tick_params(length=0, labelsize=6.8, pad=1, labelright=False, labelleft=False)
field_cbar.outline.set_linewidth(0.4)

matrix_cbar_ax = fig.add_axes(
    [
        panel_b_cbar_left,
        matrix_cbar_bottom,
        panel_b_cbar_width,
        matrix_cbar_height,
    ]
)
matrix_sm = mpl.cm.ScalarMappable(norm=b_matrix_norm, cmap=matrix_cmap)
matrix_cbar = fig.colorbar(matrix_sm, cax=matrix_cbar_ax)
matrix_cbar.set_ticks([0, 1])
matrix_cbar.ax.tick_params(length=0, labelsize=6.8, pad=1, labelright=False, labelleft=False)
matrix_cbar.outline.set_linewidth(0.4)

cbar_tick_label_x = panel_b_cbar_left + panel_b_cbar_width + 0.003
cbar_endpoint_label_offset = 0.007
for label, y_pos in [
    ("max", field_cbar_bottom + field_cbar_height - cbar_endpoint_label_offset),
    ("min", field_cbar_bottom + cbar_endpoint_label_offset),
    ("1", matrix_cbar_bottom + matrix_cbar_height - cbar_endpoint_label_offset),
    ("0", matrix_cbar_bottom + cbar_endpoint_label_offset),
]:
    fig.text(
        cbar_tick_label_x,
        y_pos,
        label,
        ha="left",
        va="center",
        fontsize=6.8,
    )

cbar_label_x = panel_b_cbar_left + panel_b_cbar_width + 0.024
matrix_cbar_label_x = panel_b_cbar_left + panel_b_cbar_width + 0.014
fig.text(
    cbar_label_x,
    field_cbar_bottom + field_cbar_height / 2,
    "Intensity",
    rotation=90,
    ha="center",
    va="center",
    fontsize=6.5,
)
fig.text(
    matrix_cbar_label_x,
    matrix_cbar_bottom + matrix_cbar_height / 2,
    r"$M_{AB}$",
    rotation=0,
    ha="left",
    va="center",
    fontsize=6.5,
)

# ---------- Panel c: N theta fitting ----------

ax_c = fig.add_axes([panel_c_left, panel_c_bottom, panel_c_width, panel_c_height])

n_pad = (float(np.nanmax(n_list)) - float(np.nanmin(n_list))) * 0.08
theta_pad = (float(np.nanmax(theta_mrad)) - float(np.nanmin(theta_mrad))) * 0.14
fit_order = np.argsort(n_list)

ax_c.scatter(
    n_list,
    theta_mrad,
    marker="v",
    s=42,
    color=data_color,
    edgecolor="none",
    linewidth=0,
    zorder=3,
    label="Simulation",
)
ax_c.plot(n_list[fit_order], theta_fit[fit_order], ls="--", lw=fit_line_width, color=fit_color, label="Linear fitting")
ax_c.set_xlabel(r"$N$")
ax_c.set_ylabel(r"$\Delta\theta$ (mrad)")
ax_c.set_xlim(float(np.nanmin(n_list)) - n_pad, float(np.nanmax(n_list)) + n_pad)
ax_c.set_ylim(float(np.nanmin(theta_mrad)) - theta_pad, float(np.nanmax(theta_mrad)) + theta_pad)
ax_c.grid(axis="y", lw=0.42, color=mist, alpha=0.78, linestyle=(0, (3.2, 2.0)))
ax_c.legend(loc="upper left", handlelength=2.8)
for spine_name in ["top", "right"]:
    ax_c.spines[spine_name].set_visible(True)
    ax_c.spines[spine_name].set_linewidth(0.75)
    ax_c.spines[spine_name].set_color(ink)

theta_sign = "+" if theta_intercept >= 0 else "-"
theta_equation = rf"$\Delta\theta={theta_slope:.3f}N {theta_sign} {abs(theta_intercept):.3f}$"
fit_angle_x0 = float(np.nanmin(n_list))
fit_angle_x1 = float(np.nanmax(n_list))
fit_angle_y0 = theta_slope * fit_angle_x0 + theta_intercept
fit_angle_y1 = theta_slope * fit_angle_x1 + theta_intercept
fit_display_start = ax_c.transData.transform((fit_angle_x0, fit_angle_y0))
fit_display_end = ax_c.transData.transform((fit_angle_x1, fit_angle_y1))
fit_text_angle = np.degrees(
    np.arctan2(
        fit_display_end[1] - fit_display_start[1],
        fit_display_end[0] - fit_display_start[0],
    )
)
fit_text_size = 7.2
theta_equation_x = 10.2
ax_c.text(
    theta_equation_x,
    theta_slope * theta_equation_x + theta_intercept - 0.18,
    theta_equation,
    rotation=fit_text_angle,
    rotation_mode="anchor",
    ha="center",
    va="center",
    fontsize=fit_text_size,
    fontfamily="sans-serif",
    color=fit_color,
)
r2_text_x = 12.4
ax_c.text(
    r2_text_x,
    theta_slope * r2_text_x + theta_intercept + 0.15,
    rf"$R^2={theta_r2:.3f}$",
    rotation=fit_text_angle,
    rotation_mode="anchor",
    ha="center",
    va="center",
    fontsize=fit_text_size,
    fontfamily="sans-serif",
    color=fit_color,
)
# ---------- Panel d: distance variation and N.A. control ----------

ax_d = fig.add_axes([panel_d_left, panel_d_bottom, panel_d_width, panel_d_height])

if len(distance_um) != len(na_list):
    raise ValueError("distance1[200-500 um] and NA_list must have the same number of points.")

d_y_all = np.r_[distance_ct, na_ct]
d_y_pad = (float(np.nanmax(d_y_all)) - float(np.nanmin(d_y_all))) * 0.14

ax_d.scatter(
    distance_um,
    distance_ct,
    marker="o",
    s=40,
    facecolor=distance_color,
    edgecolor="none",
    linewidth=0,
    label=r"Varying $d$",
    zorder=3,
)
ax_d.scatter(
    distance_um,
    na_ct,
    marker="v",
    s=40,
    facecolor=na_color,
    edgecolor="none",
    linewidth=0,
    label="N.A. control",
    zorder=4,
)
ax_d.set_xlim(185, 515)
ax_d.set_ylim(float(np.nanmin(d_y_all)) - d_y_pad, float(np.nanmax(d_y_all)) + d_y_pad)
ax_d.set_xlabel(r"$d$ ($\mu$m)")
ax_d.set_ylabel("Mean crosstalk")
ax_d.set_xticks(distance_um)
ax_d.grid(False)
ax_d.legend(loc="lower right", handlelength=1.0)
ax_d.spines["right"].set_visible(True)
ax_d.spines["right"].set_linewidth(0.75)
ax_d.spines["right"].set_color(ink)

ax_d_top = ax_d.twiny()
ax_d_top.set_xlim(ax_d.get_xlim())
ax_d_top.set_xticks(distance_um)
ax_d_top.set_xticklabels([f"{value:.2f}" for value in na_list])
ax_d_top.set_xlabel("N.A.")
ax_d_top.tick_params(axis="x", labelsize=7, length=2.5)
ax_d_top.spines["top"].set_visible(True)
ax_d_top.spines["top"].set_linewidth(0.75)
ax_d_top.spines["top"].set_color(ink)
ax_d_top.spines["bottom"].set_visible(False)
ax_d_top.spines["right"].set_visible(True)
ax_d_top.spines["right"].set_linewidth(0.75)
ax_d_top.spines["right"].set_color(ink)

# ---------- Save ----------

fig.savefig(BASE_DIR / "Fig2_8.svg", format="svg", dpi=1200)
plt.close(fig)
