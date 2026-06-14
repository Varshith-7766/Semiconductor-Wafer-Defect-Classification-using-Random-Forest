"""
Script to generate LinkedIn post images for the project.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_code_snippet_image():
    """Create a code snippet image for LinkedIn."""
    code_lines = [
        ("# Feature Engineering - Key Logic", "#2E86AB", 0.95),
        ("", "#2E86AB", 0.95),
        ("def extract_spatial_features(wafer_map):", "#A23B72", 0.95),
        ("    # Center failure density", "#2E86AB", 0.95),
        ("    center_region = failure_mask[", "#2E86AB", 0.95),
        ("        center_h - margin_h : center_h + margin_h,", "#2E86AB", 0.95),
        ("        center_w - margin_w : center_w + margin_w]", "#2E86AB", 0.95),
        ("    center_density = np.mean(center_region)", "#2E86AB", 0.95),
        ("", "#2E86AB", 0.95),
        ("    # Shannon entropy of failure distribution", "#2E86AB", 0.95),
        ("    failure_entropy = (row_entropy + col_entropy) / 2", "#2E86AB", 0.95),
        ("", "#2E86AB", 0.95),
        ("    # Connected component analysis", "#2E86AB", 0.95),
        ("    labeled, num_clusters = ndimage.label(failure_mask)", "#2E86AB", 0.95),
        ("    largest_cluster = np.max(cluster_sizes)", "#2E86AB", 0.95),
        ("", "#2E86AB", 0.95),
        ("    return {", "#F18F01", 0.95),
        ("        'center_failure_density': center_density,", "#F18F01", 0.95),
        ("        'failure_entropy': failure_entropy,", "#F18F01", 0.95),
        ("        'failure_cluster_count': num_clusters,", "#F18F01", 0.95),
        ("        'largest_cluster_size': largest_cluster", "#F18F01", 0.95),
        ("    }", "#F18F01", 0.95),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#1E1E1E")
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(code_lines) + 1)
    ax.axis("off")

    # Title bar
    bar = mpatches.FancyBboxPatch(
        (0, len(code_lines)), 1, 0.8,
        boxstyle="round,pad=0.02",
        facecolor="#2D2D2D",
        edgecolor="none"
    )
    ax.add_patch(bar)
    ax.text(0.02, len(code_lines) + 0.4, "feature_engineering.py",
            color="#CCCCCC", fontsize=11, fontfamily="monospace", va="center")

    # Traffic light dots
    for i, color in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        circle = plt.Circle((0.92 + i * 0.03, len(code_lines) + 0.4), 0.015,
                           facecolor=color, edgecolor="none")
        ax.add_patch(circle)

    for i, (line, color, alpha) in enumerate(code_lines):
        y = len(code_lines) - i - 0.5
        ax.text(0.02, y, line, color=color, fontsize=10,
                fontfamily="monospace", va="center", alpha=alpha)

    plt.tight_layout(pad=0.5)
    plt.savefig("outputs/linkedin_code_snippet.png", dpi=150, bbox_inches="tight",
                facecolor="#1E1E1E", edgecolor="none")
    plt.close()
    print("Saved: outputs/linkedin_code_snippet.png")


def create_results_summary():
    """Create a results summary image for LinkedIn."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.92, "Semiconductor Wafer Defect Classification",
            fontsize=22, fontweight="bold", color="#58A6FF",
            ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.84, "Random Forest with Feature Engineering",
            fontsize=14, color="#8B949E", ha="center", va="top", transform=ax.transAxes)

    # Metrics boxes
    metrics = [
        ("Accuracy", "94.98%", "#2ECC71"),
        ("Precision", "94.33%", "#3498DB"),
        ("Recall", "94.98%", "#E67E22"),
        ("F1 Score", "94.12%", "#9B59B6"),
    ]

    for i, (name, value, color) in enumerate(metrics):
        x = 0.12 + i * 0.22
        # Box
        box = mpatches.FancyBboxPatch(
            (x - 0.08, 0.60), 0.18, 0.18,
            boxstyle="round,pad=0.02",
            facecolor="#161B22",
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + 0.01, 0.74, value, fontsize=20, fontweight="bold",
                color=color, ha="center", va="center", transform=ax.transAxes)
        ax.text(x + 0.01, 0.64, name, fontsize=11, color="#8B949E",
                ha="center", va="center", transform=ax.transAxes)

    # Key highlights
    highlights = [
        "21 engineered features from wafer maps",
        "9 defect classes classified",
        "811K samples processed",
        "Model: Random Forest (500 trees)",
    ]

    for i, text in enumerate(highlights):
        ax.text(0.12, 0.48 - i * 0.07, f"  {text}", fontsize=12, color="#C9D1D9",
                ha="left", va="center", transform=ax.transAxes,
                fontfamily="monospace")

    # GitHub link
    ax.text(0.5, 0.10, "GitHub: Varshith-7766/Semiconductor-Wafer-Defect-Classification",
            fontsize=11, color="#58A6FF", ha="center", va="center",
            transform=ax.transAxes, style="italic")

    plt.tight_layout(pad=0.5)
    plt.savefig("outputs/linkedin_results_summary.png", dpi=150, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    plt.close()
    print("Saved: outputs/linkedin_results_summary.png")


def create_project_architecture():
    """Create project architecture image for LinkedIn."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(5, 6.6, "Project Architecture", fontsize=20, fontweight="bold",
            color="#58A6FF", ha="center", va="center")

    # Flow boxes
    steps = [
        (1, 5.2, "Load Data\n(LSWMD.pkl)", "#2ECC71"),
        (3, 5.2, "Clean &\nFilter", "#E67E22"),
        (5, 5.2, "Feature\nEngineering", "#9B59B6"),
        (7, 5.2, "Train/Test\nSplit", "#3498DB"),
        (9, 5.2, "Random\nForest", "#E74C3C"),
    ]

    for x, y, text, color in steps:
        box = mpatches.FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.05",
            facecolor="#161B22",
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, color="white",
                ha="center", va="center", fontweight="bold")

    # Arrows
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(steps[i+1][0] - 0.8, steps[i+1][1]),
                   xytext=(steps[i][0] + 0.8, steps[i][1]),
                   arrowprops=dict(arrowstyle="->", color="#58A6FF", lw=2))

    # Output boxes
    outputs = [
        (1.5, 3.2, "Confusion\nMatrix", "#2ECC71"),
        (4, 3.2, "Feature\nImportance", "#E67E22"),
        (6.5, 3.2, "Classification\nReport", "#9B59B6"),
        (9, 3.2, "Saved Model\n(model.pkl)", "#3498DB"),
    ]

    for x, y, text, color in outputs:
        box = mpatches.FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.05",
            facecolor="#161B22",
            edgecolor=color,
            linewidth=2,
            linestyle="--"
        )
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=10, color="white",
                ha="center", va="center")

    # Labels
    ax.text(5, 4.2, "Outputs", fontsize=14, fontweight="bold",
            color="#8B949E", ha="center", va="center")

    # Arrows from RF to outputs
    for x, _, _, _ in outputs:
        ax.annotate("", xy=(x, 3.6), xytext=(9, 4.8),
                   arrowprops=dict(arrowstyle="->", color="#58A6FF", lw=1.5,
                                  connectionstyle="arc3,rad=0.1"))

    plt.tight_layout(pad=0.5)
    plt.savefig("outputs/linkedin_architecture.png", dpi=150, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    plt.close()
    print("Saved: outputs/linkedin_architecture.png")


if __name__ == "__main__":
    create_code_snippet_image()
    create_results_summary()
    create_project_architecture()
    print("\nAll LinkedIn images generated!")
