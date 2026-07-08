import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .heuristics import bounding_circle

# --- theme (dataviz skill palette: fixed categorical order, light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

METHOD_COLORS = {
    "exact": "#2a78d6",       # categorical slot 1 (blue)
    "nn2opt": "#1baf7a",      # categorical slot 2 (aqua)
    "angular": "#eda100",     # categorical slot 3 (yellow)
    "shrinkwrap": "#4a3aa7",  # categorical slot 5 (violet)
    "orbit": "#e34948",       # categorical slot 6 (red)
}
METHOD_LABELS = {
    "exact": "Exact (Held–Karp)",
    "nn2opt": "Nearest-Neighbor + 2-opt",
    "angular": "Angular Sort (wedge)",
    "shrinkwrap": "Shrink-Wrap (vacuum bag)",
    "orbit": "Orbit & Recenter",
}

def _style_ax(ax, equal=True):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if equal:
        ax.set_aspect("equal")
    ax.grid(False)


def _draw_tour(ax, points, tour, color, point_color=INK):
    pts = points[tour + [tour[0]]]
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=26, color=point_color, zorder=3, edgecolors=SURFACE, linewidths=0.8)
    ax.scatter(points[tour[0], 0], points[tour[0], 1], s=90, facecolors="none",
               edgecolors=color, linewidths=2, zorder=4)


def plot_static_comparison(points, results, save_path=None, title=None, suptitle_note=None):
    """results: ordered dict of key -> {'tour', 'length', 'time'}"""
    keys = list(results.keys())
    best_len = min(r["length"] for r in results.values())

    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.8), facecolor=SURFACE)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        r = results[key]
        color = METHOD_COLORS.get(key, INK)
        _style_ax(ax)
        _draw_tour(ax, points, r["tour"], color)
        pct = 100.0 * (r["length"] / best_len - 1.0)
        pct_str = "best" if pct < 1e-6 else f"+{pct:.1f}% vs best"
        ax.set_title(
            f"{METHOD_LABELS.get(key, key)}\nlength {r['length']:.1f}  ·  {pct_str}\n"
            f"{r['time'] * 1000:.2f} ms",
            fontsize=10.5, color=INK, loc="left",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=13, color=INK, y=1.04)
    if suptitle_note:
        fig.text(0.5, -0.02, suptitle_note, ha="center", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# Approach 1 animation: rotating radial sweep around the bounding circle
# ---------------------------------------------------------------------------

def animate_angular_sweep(points, save_path, clockwise=True, fps=20, frames=180):
    c, r = bounding_circle(points)
    vecs = points - c
    raw_angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    angles = (-raw_angles if clockwise else raw_angles) % (2 * np.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    _style_ax(ax)
    pad = r * 0.25
    ax.set_xlim(c[0] - r - pad, c[0] + r + pad)
    ax.set_ylim(c[1] - r - pad, c[1] + r + pad)
    ax.set_xticks([])
    ax.set_yticks([])

    circle = patches.Circle(c, r, fill=False, linestyle="--", linewidth=1.3, edgecolor=INK_MUTED, zorder=1)
    ax.add_patch(circle)
    ax.scatter(*c, marker="+", s=60, color=INK_SECONDARY, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=26, color=INK, zorder=3, edgecolors=SURFACE, linewidths=0.8)

    (sweep_line,) = ax.plot([], [], color=METHOD_COLORS["angular"], linewidth=1.6, alpha=0.85, zorder=4)
    (path_line,) = ax.plot([], [], color=METHOD_COLORS["angular"], linewidth=2.4, zorder=5)
    ax.set_title("Angular Sort: sweep the bounding circle, collect points in order", fontsize=11, color=INK)

    def update(frame):
        t = frame / frames
        theta = t * 2 * np.pi
        end = c + r * 1.05 * np.array([np.cos(-theta if clockwise else theta), np.sin(-theta if clockwise else theta)])
        sweep_line.set_data([c[0], end[0]], [c[1], end[1]])

        passed = order[sorted_angles <= theta + 1e-9]
        if len(passed) > 0:
            pts = points[passed]
            path_line.set_data(pts[:, 0], pts[:, 1])
        else:
            path_line.set_data([], [])
        if frame == frames:
            closed = points[np.append(order, order[0])]
            path_line.set_data(closed[:, 0], closed[:, 1])
        return sweep_line, path_line

    anim = FuncAnimation(fig, update, frames=frames + 1, interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Approach 2 animation: circle shrinks uniformly, contacting points in order
# of decreasing distance from the centroid; each new point is spliced into
# the loop at the edge where it forms the shallowest triangle.
# ---------------------------------------------------------------------------

def animate_shrink_wrap(points, tour, trace, save_path, fps=1.5, hold_frames=1):
    n = len(points)
    c, r = bounding_circle(points)
    dists = np.linalg.norm(points - c, axis=1)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    _style_ax(ax)
    pad = r * 0.25
    ax.set_xlim(c[0] - r - pad, c[0] + r + pad)
    ax.set_ylim(c[1] - r - pad, c[1] + r + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Shrink-Wrap: circle contacts points far-to-near, each snaps in at its\nshallowest-triangle edge",
                 fontsize=10.5, color=INK)

    shrink_circle = patches.Circle(c, r * 1.08, fill=False, linestyle="--", linewidth=1.3,
                                    edgecolor=INK_MUTED, zorder=1)
    ax.add_patch(shrink_circle)

    (tri_line,) = ax.plot([], [], color=METHOD_COLORS["shrinkwrap"], linewidth=1.1,
                          linestyle=":", alpha=0.7, zorder=3)
    scat = ax.scatter(points[:, 0], points[:, 1], s=26, color=INK_MUTED, zorder=4,
                       edgecolors=SURFACE, linewidths=0.8)
    (path_line,) = ax.plot([], [], color=METHOD_COLORS["shrinkwrap"], linewidth=2.2, zorder=5)

    n_frames = len(trace) * hold_frames

    def update(frame):
        idx = min(len(trace) - 1, frame // hold_frames)
        entry = trace[idx]
        path = entry["path"]

        closed = points[path + [path[0]]]
        path_line.set_data(closed[:, 0], closed[:, 1])

        revealed_set = set(path)
        sizes = [90 if i == entry["inserted"] else 26 for i in range(n)]
        colors = [METHOD_COLORS["shrinkwrap"] if i in revealed_set else INK_MUTED for i in range(n)]
        scat.set_color(colors)
        scat.set_sizes(sizes)

        if entry["edge"] is not None:
            a, b = entry["edge"]
            tri = points[[a, entry["inserted"], b, a]]
            tri_line.set_data(tri[:, 0], tri[:, 1])
            radius = dists[entry["inserted"]]
        else:
            tri_line.set_data([], [])
            radius = r
        shrink_circle.set_radius(radius * 1.08)

        return scat, path_line, shrink_circle, tri_line

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Approach 3 animation: circle orbits and picks off the next point in its
# forward half; after every pick the circle recenters on whatever's left.
# ---------------------------------------------------------------------------

def animate_orbit_recenter(points, tour, trace, save_path, fps=2):
    n = len(points)
    c0, r0 = bounding_circle(points)
    pad = r0 * 0.25

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    _style_ax(ax)
    ax.set_xlim(c0[0] - r0 - pad, c0[0] + r0 + pad)
    ax.set_ylim(c0[1] - r0 - pad, c0[1] + r0 + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Orbit & Recenter: circle picks off the next forward point, then\nrecenters on what's left",
                 fontsize=10.5, color=INK)

    circle = patches.Circle(c0, r0, fill=False, linestyle="--", linewidth=1.3,
                            edgecolor=INK_MUTED, zorder=1)
    ax.add_patch(circle)
    center_mark = ax.scatter(*c0, marker="+", s=60, color=INK_SECONDARY, zorder=2)

    scat = ax.scatter(points[:, 0], points[:, 1], s=26, color=INK_MUTED, zorder=4,
                       edgecolors=SURFACE, linewidths=0.8)
    (path_line,) = ax.plot([], [], color=METHOD_COLORS["orbit"], linewidth=2.2, zorder=5)
    (walker,) = ax.plot([], [], marker="o", markersize=7, color=METHOD_COLORS["orbit"],
                        markerfacecolor="none", markeredgewidth=2, zorder=6)

    def update(frame):
        entry = trace[frame]
        circle.set_center(entry["center"])
        circle.set_radius(entry["radius"])
        center_mark.set_offsets([entry["center"]])
        walker.set_data([entry["cur_pos"][0]], [entry["cur_pos"][1]])

        path = entry["path"]
        if path:
            pts = points[path]
            path_line.set_data(pts[:, 0], pts[:, 1])
        else:
            path_line.set_data([], [])
        if entry is trace[-1] and len(path) == n:
            closed = points[path + [path[0]]]
            path_line.set_data(closed[:, 0], closed[:, 1])

        placed = set(path)
        colors = [METHOD_COLORS["orbit"] if i in placed else INK_MUTED for i in range(n)]
        scat.set_color(colors)

        return circle, center_mark, scat, path_line, walker

    anim = FuncAnimation(fig, update, frames=len(trace), interval=1000 / fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Benchmark plots
# ---------------------------------------------------------------------------

def plot_benchmark(records, save_path, methods=("exact", "nn2opt", "angular", "shrinkwrap", "orbit")):
    ns = sorted(set(r["n"] for r in records))
    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax in (ax_q, ax_t):
        _style_ax(ax, equal=False)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)

    for m in methods:
        color = METHOD_COLORS[m]
        q_med, q_lo, q_hi, t_med = [], [], [], []
        valid_ns = []
        for n in ns:
            vals = [r["ratio_to_best"] for r in records if r["n"] == n and r["method"] == m]
            times = [r["time"] for r in records if r["n"] == n and r["method"] == m]
            if not vals:
                continue
            valid_ns.append(n)
            q_med.append(np.median(vals))
            q_lo.append(np.min(vals))
            q_hi.append(np.max(vals))
            t_med.append(np.median(times))
        if not valid_ns:
            continue
        ax_q.plot(valid_ns, q_med, "-o", color=color, label=METHOD_LABELS[m], markersize=4, linewidth=2, zorder=3)
        ax_q.fill_between(valid_ns, q_lo, q_hi, color=color, alpha=0.12, zorder=2)
        ax_t.plot(valid_ns, t_med, "-o", color=color, label=METHOD_LABELS[m], markersize=4, linewidth=2, zorder=3)

    ax_q.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax_q.set_xlabel("points (n)", color=INK_SECONDARY, fontsize=9)
    ax_q.set_ylabel("tour length ÷ best-known length", color=INK_SECONDARY, fontsize=9)
    ax_q.set_title("Solution quality (lower is better, 1.0 = best found)", fontsize=10.5, color=INK, loc="left")

    ax_t.set_yscale("log")
    ax_t.set_xlabel("points (n)", color=INK_SECONDARY, fontsize=9)
    ax_t.set_ylabel("wall-clock time (s, log scale)", color=INK_SECONDARY, fontsize=9)
    ax_t.set_title("Runtime growth", fontsize=10.5, color=INK, loc="left")

    handles, labels = ax_q.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return save_path
