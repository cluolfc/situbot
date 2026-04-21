#!/usr/bin/env python3
"""Visualization helpers for SituBot arrangements."""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def plot_arrangement(placements: List[Dict],
                     workspace_bounds: Dict,
                     title: str = "Arrangement",
                     object_catalog: Optional[Dict] = None,
                     save_path: Optional[str] = None,
                     show: bool = True):
    """Plot a top-down view of object arrangement on the table.

    Args:
        placements: List of dicts with name, x, y, role keys.
        workspace_bounds: Dict with x_min, x_max, y_min, y_max.
        title: Plot title (typically the situation description).
        object_catalog: Dict of name → object info (for dimensions).
        save_path: If set, save figure to this path.
        show: Whether to display the plot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    b = workspace_bounds

    # Draw table boundary
    table_w = b["y_max"] - b["y_min"]
    table_d = b["x_max"] - b["x_min"]
    table_rect = patches.Rectangle(
        (b["y_min"], b["x_min"]), table_w, table_d,
        linewidth=2, edgecolor="brown", facecolor="wheat", alpha=0.3,
    )
    ax.add_patch(table_rect)

    # Color by role
    role_colors = {
        "prominent": "#e74c3c",    # red
        "accessible": "#3498db",   # blue
        "peripheral": "#95a5a6",   # gray
        "remove": "#2c3e50",       # dark
        "": "#7f8c8d",             # default gray
    }

    for p in placements:
        name = p["name"]
        x, y = p["x"], p["y"]
        role = p.get("role", "")
        color = role_colors.get(role, "#7f8c8d")

        # Get object dimensions if available
        if object_catalog and name in object_catalog:
            dims = object_catalog[name].get("dimensions", {})
            w = dims.get("w", 0.08)
            d = dims.get("d", 0.08)
        else:
            w, d = 0.08, 0.08

        # Draw object as rectangle (y → horizontal, x → vertical in plot)
        rect = patches.Rectangle(
            (y - w / 2, x - d / 2), w, d,
            linewidth=1, edgecolor=color, facecolor=color, alpha=0.5,
        )
        ax.add_patch(rect)
        ax.annotate(name, (y, x), fontsize=7, ha="center", va="center",
                     fontweight="bold")

    # Mark person's side
    ax.annotate("← Person sits here", (0, b["x_min"] - 0.02),
                fontsize=10, ha="center", color="green")

    ax.set_xlim(b["y_min"] - 0.05, b["y_max"] + 0.05)
    ax.set_ylim(b["x_min"] - 0.05, b["x_max"] + 0.05)
    ax.set_xlabel("Y (left ← → right)")
    ax.set_ylabel("X (close → far)")
    ax.set_title(title, fontsize=10, wrap=True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    for role, color in role_colors.items():
        if role:
            ax.plot([], [], "s", color=color, alpha=0.5, label=role, markersize=10)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved arrangement plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
