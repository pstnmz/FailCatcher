import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import numpy as np
from medmnist import BreastMNIST

# ---------------- Config ----------------
N_SAMPLES = 8
DPI = 150
FIGSIZE = (14, 6)
MS_PER_FRAME = 1000       # 🕐 much slower (2 seconds per frame)
ACCEPT_THRESHOLD = 0.75
CMAP = plt.cm.RdYlGn      # green = high confidence

# ---------------- Data ----------------
data = BreastMNIST(split='test', download=True, size=224)
imgs, labels = data.imgs[:N_SAMPLES], data.labels[:N_SAMPLES].flatten()

# ---------------- Helpers ----------------
def draw_model_icon(ax, cx=0.5, cy=0.5):
    """Minimal circle + 3-node graph."""
    circ = patches.Circle((cx, cy), 0.18, fill=False, lw=2, transform=ax.transAxes)
    ax.add_patch(circ)
    p1 = (cx, cy + 0.09)
    p2 = (cx - 0.09, cy - 0.07)
    p3 = (cx + 0.09, cy - 0.07)
    for a, b in [(p1, p2), (p2, p3), (p3, p1)]:
        ax.add_line(plt.Line2D([a[0], b[0]], [a[1], b[1]], transform=ax.transAxes, lw=2, color='black'))
    for p in [p1, p2, p3]:
        ax.add_patch(patches.Circle(p, 0.012, color="black", transform=ax.transAxes))

def add_bottom_colorbar(ax_right, confidence, cmap=CMAP):
    """Static horizontal colorbar INSIDE the right panel, with a tick at confidence."""
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax_right.inset_axes([0.00, 0.02, 1.00, 0.14])
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_label("Confidence")
    cax.axvline(confidence, color='black', linewidth=2)

# ---------------- Build frames ----------------
frames = []

for i, (img, label) in enumerate(zip(imgs, labels)):
    pred_text = "Malignant" if label else "Benign"
    confidence = float(np.random.uniform(0.4, 0.99))
    decision = "ACCEPT" if confidence >= ACCEPT_THRESHOLD else "REVIEW"
    decision_color = (0.10, 0.6, 0.10) if decision == "ACCEPT" else (0.9, 0.55, 0.1)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[0.16, 0.84],
                          width_ratios=[0.42, 0.20, 0.38])

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.6, "AI Failure Detection via Uncertainty Quantification",
                  fontsize=18, weight="bold", ha="center", va="center")

    # Left: image
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(img, cmap='gray')
    ax_img.axis("off")

    # Center: model
    ax_model = fig.add_subplot(gs[1, 1])
    ax_model.axis("off")
    draw_model_icon(ax_model)

    # Right: text + colorbar
    ax_uq = fig.add_subplot(gs[1, 2])
    ax_uq.axis("off")

    # Prediction aligned with model center
    ax_uq.text(0.02, 0.50, f"Prediction: {pred_text}", fontsize=14,
               ha="left", va="center", transform=ax_uq.transAxes)

    # Confidence and decision just above colorbar
    ax_uq.text(0.02, 0.28, f"Confidence: {confidence:.2f}", fontsize=12,
               ha="left", va="center", transform=ax_uq.transAxes)
    ax_uq.text(0.02, 0.20, f"Decision: {decision}", fontsize=12, weight="bold",
               ha="left", va="center", transform=ax_uq.transAxes, color=decision_color)

    # Static colorbar (inside panel)
    add_bottom_colorbar(ax_uq, confidence=confidence, cmap=CMAP)

    # Render frame
    fig.canvas.draw()
    frames.append(Image.fromarray(np.array(fig.canvas.renderer.buffer_rgba())))
    plt.close(fig)

# Save GIF (slow version)
frames[0].save(
    "uq_failure_detection_centered.gif",
    save_all=True,
    append_images=frames[1:],
    duration=MS_PER_FRAME,
    loop=0,
    optimize=True,
    quality=95
)
