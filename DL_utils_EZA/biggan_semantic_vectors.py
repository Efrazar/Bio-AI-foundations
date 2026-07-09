"""
BigGAN Latent Space Exploration — Semantic Vectors
====================================================
A step-by-step pipeline to:
  1. Generate an 8×5 grid of random images to visually explore the latent space
  2. Label images by attribute to define positive / negative examples
  3. Compute a semantic vector from those labels
  4. Use the semantic vector to manipulate any image in a controlled, interpretable way

Conceptual foundation
---------------------
BigGAN maps a 128-dimensional noise vector (z) + a class embedding (y)
to a 256×256 image. Directions in that 128-dim z-space are not random —
they encode meaningful visual attributes (pose, lighting, background,
facial expression, etc.). This is because the GAN's discriminator forces
the generator to organise z-space so that nearby z vectors produce
visually similar images.

The "semantic vector" technique (originally from GAN vector arithmetic,
popularised by Radford et al. 2015 and refined in InterFaceGAN, GANSpace,
etc.) finds those directions by:

    semantic_vector = mean(z_positive_examples)
                    - mean(z_negative_examples)

This difference vector points, in z-space, in the direction that ADDS
the labelled attribute. Scaling it with a coefficient α lets you dial
the attribute up or down on any image:

    z_manipulated = z_original + α × semantic_vector

The same principle underlies modern interpretability tools for diffusion
models and is directly relevant to biological image analysis — where you
might ask "what direction in latent space encodes cell confluence?" or
"which z dimension correlates with nucleus size?".

Hardware note
-------------
All generation is done under torch.no_grad() with AMP autocast, keeping
peak VRAM well under the 10.57 GB budget of the RTX 2080 Ti. Images are
generated one at a time (batch_size=1) intentionally — we store z vectors
in CPU RAM (31 GB available) and only push one tensor to the GPU at a time.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pytorch_pretrained_biggan import (
    BigGAN, truncated_noise_sample, one_hot_from_names, convert_to_images
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate an 8×5 grid of random images
# ─────────────────────────────────────────────────────────────────────────────

def generate_image_grid(model, class_name, device,
                         n_rows=8, n_cols=5,
                         truncation=0.4,
                         dim_z=128,
                         seed=42):
    """
    Generate an (n_rows × n_cols) grid of random BigGAN images for one class.

    Each image gets its own independent random z vector, sampled from a
    truncated normal distribution. The truncation parameter controls
    "how far out" in z-space we sample:
        - Low  (0.2–0.5) → safe, high-quality images, less variety
        - High (0.7–1.0) → more variety, more risk of artifacts

    Parameters
    ----------
    model       : BigGAN — loaded pretrained model already on device
    class_name  : str    — any ImageNet class name (e.g. 'golden retriever')
    device      : str    — 'cuda' or 'cpu'
    n_rows      : int    — grid rows
    n_cols      : int    — grid columns
    truncation  : float  — truncation value for noise sampling
    dim_z       : int    — z dimension (128 for all BigGAN variants)
    seed        : int    — random seed for reproducibility

    Returns
    -------
    frames  : list[PIL.Image] — generated images (length = n_rows * n_cols)
    z_bank  : torch.Tensor   — all z vectors, shape (n_rows*n_cols, dim_z)
                               stored on CPU for later semantic vector math
    """
    np.random.seed(seed)
    n_images = n_rows * n_cols

    y = torch.from_numpy(
        one_hot_from_names([class_name], batch_size=1)
    ).to(device)

    frames = []
    z_list = []

    model.eval()
    with torch.no_grad():
        for i in range(n_images):
            z_np = truncated_noise_sample(batch_size=1, dim_z=dim_z,
                                           truncation=truncation)
            z = torch.from_numpy(z_np).to(device)

            with torch.amp.autocast(device_type="cuda"):
                out = model(z, y, truncation=truncation)

            frames.extend(convert_to_images(out.cpu()))
            z_list.append(torch.from_numpy(z_np))   # store on CPU

    # Stack into (N, dim_z) for easy indexing later
    z_bank = torch.cat(z_list, dim=0)

    # ── Display grid with index numbers so you know which image is which
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.2 * n_cols, 2.4 * n_rows))
    fig.patch.set_facecolor('#1a1a2e')

    for i, (ax, frame) in enumerate(zip(axes.flatten(), frames)):
        ax.imshow(frame)
        ax.set_title(f"[{i}]", fontsize=9, color='#e0e0e0',
                     fontweight='bold', pad=2)
        ax.axis("off")

    plt.suptitle(
        f"BigGAN latent space — {class_name}  |  "
        f"{n_images} random z samples  |  truncation={truncation}\n"
        "Note the image index [n] — you will use these numbers to label attributes",
        fontsize=11, color='white', y=1.01
    )
    plt.tight_layout()
    plt.show()

    print(f"Generated {n_images} images.")
    print(f"z_bank shape: {z_bank.shape}  — ({n_images} vectors × {dim_z} dims)")
    print("\nNext step: pick image indices for your attribute of interest.")
    print("  pos_indices = [i, j, k, ...]   # images that HAVE the attribute")
    print("  neg_indices = [x, y, z, ...]   # images that DO NOT")

    return frames, z_bank


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Label the latent dimensions (interactive, done in notebook)
# ─────────────────────────────────────────────────────────────────────────────

def preview_labeled_images(frames, pos_indices, neg_indices,
                            attribute_name="attribute"):
    """
    Preview your labeling choices before computing the semantic vector.

    Shows positives (green border) and negatives (red border) side by side
    so you can verify your selections visually before committing.

    Parameters
    ----------
    frames          : list[PIL.Image] — from generate_image_grid()
    pos_indices     : list[int]       — grid indices you labelled as POSITIVE
    neg_indices     : list[int]       — grid indices you labelled as NEGATIVE
    attribute_name  : str             — label for the title
    """
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    n_cols = max(n_pos, n_neg)

    fig, axes = plt.subplots(2, n_cols,
                              figsize=(2.5 * n_cols, 5.5))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Row 0: positives
    for col in range(n_cols):
        ax = axes[0, col]
        if col < n_pos:
            ax.imshow(frames[pos_indices[col]])
            for spine in ax.spines.values():
                spine.set_edgecolor('#2ecc71')
                spine.set_linewidth(3)
            ax.set_title(f"[{pos_indices[col]}]  ✔ positive",
                          fontsize=8, color='#2ecc71', pad=3)
        ax.axis("off")

    # ── Row 1: negatives
    for col in range(n_cols):
        ax = axes[1, col]
        if col < n_neg:
            ax.imshow(frames[neg_indices[col]])
            for spine in ax.spines.values():
                spine.set_edgecolor('#e74c3c')
                spine.set_linewidth(3)
            ax.set_title(f"[{neg_indices[col]}]  ✘ negative",
                          fontsize=8, color='#e74c3c', pad=3)
        ax.axis("off")

    axes[0, 0].set_ylabel("POSITIVE", fontsize=10, color='#2ecc71',
                            fontweight='bold', rotation=90, labelpad=6)
    axes[1, 0].set_ylabel("NEGATIVE", fontsize=10, color='#e74c3c',
                            fontweight='bold', rotation=90, labelpad=6)

    plt.suptitle(
        f"Label preview — attribute: '{attribute_name}'\n"
        f"{n_pos} positive examples   |   {n_neg} negative examples",
        fontsize=11, color='white'
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Compute the semantic vector
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_vector(z_bank, pos_indices, neg_indices):
    """
    Compute the semantic direction vector for a visual attribute.

    The semantic vector is the difference between the mean z of images
    that HAVE the attribute and the mean z of images that DON'T:

        semantic_vector = mean(z_positive) - mean(z_negative)

    Why this works
    --------------
    The GAN was trained so that nearby z vectors produce visually similar
    images. If a cluster of z vectors reliably produces images WITH an
    attribute (e.g. "looking left"), the average of that cluster points
    toward the region of z-space that encodes that attribute.

    Subtracting the negative mean removes the "baseline" and isolates
    the pure direction of the attribute — similar to how word2vec encodes
    semantic relationships as vector differences (king - man + woman ≈ queen).

    The more labeled examples you provide, the more robust the direction.
    As a rule of thumb: ≥5 positives and ≥5 negatives produce meaningful
    results; ≥15 each produce reliable ones.

    Parameters
    ----------
    z_bank      : torch.Tensor — shape (N, dim_z), from generate_image_grid()
    pos_indices : list[int]    — indices of POSITIVE examples
    neg_indices : list[int]    — indices of NEGATIVE examples

    Returns
    -------
    semantic_vec : torch.Tensor — shape (1, dim_z), unit-normalised direction
                                  stored on CPU; send to device before use
    stats        : dict         — diagnosis info (norms, separation)
    """
    z_pos = z_bank[pos_indices]     # (n_pos, dim_z)
    z_neg = z_bank[neg_indices]     # (n_neg, dim_z)

    mean_pos = z_pos.mean(dim=0)    # (dim_z,)
    mean_neg = z_neg.mean(dim=0)    # (dim_z,)

    raw_vec = mean_pos - mean_neg   # (dim_z,)

    # Normalise to unit length so α is comparable across different attributes
    # (raw vectors from different attributes can have very different magnitudes,
    # making α values incomparable; unit normalisation fixes this)
    norm = raw_vec.norm()
    semantic_vec = (raw_vec / norm).unsqueeze(0)   # (1, dim_z)

    # ── Diagnostic: cosine separation between pos and neg clusters
    cos_sim = torch.nn.functional.cosine_similarity(
        mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)
    ).item()

    stats = {
        "n_positive"       : len(pos_indices),
        "n_negative"       : len(neg_indices),
        "raw_vector_norm"  : norm.item(),
        "pos_neg_cosine"   : cos_sim,
        "cluster_separation": (1 - cos_sim),   # 0 = identical, 2 = opposite
    }

    print("── Semantic vector computed ──────────────────")
    print(f"  Positive examples     : {stats['n_positive']}")
    print(f"  Negative examples     : {stats['n_negative']}")
    print(f"  Raw vector norm       : {stats['raw_vector_norm']:.4f}")
    print(f"  Cluster separation    : {stats['cluster_separation']:.4f}  "
          f"(higher = better-separated attribute, range 0–2)")

    if stats["cluster_separation"] < 0.1:
        print("  ⚠ Low separation — try adding more / cleaner labels")
    elif stats["cluster_separation"] > 0.5:
        print("  ✔ Good cluster separation — attribute is well-encoded in z")

    return semantic_vec, stats


def visualize_semantic_vector(semantic_vec, dim_z=128):
    """
    Bar chart of the semantic vector's component weights.

    Each bar represents one of the 128 z dimensions. Tall positive bars
    are the dimensions the model activates most to produce the attribute;
    tall negative bars are dimensions the model suppresses.

    This is the closest thing to "labeling the latent dimensions" —
    you can see WHICH dimensions carry the attribute the most, even though
    BigGAN's z-dimensions are not individually interpretable (unlike PCA
    directions in GANSpace), this gives useful intuition about sparsity.

    Parameters
    ----------
    semantic_vec : torch.Tensor — shape (1, dim_z), from compute_semantic_vector()
    """
    vec_np = semantic_vec.squeeze().numpy()

    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vec_np]

    fig, ax = plt.subplots(figsize=(16, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d1117')

    ax.bar(range(dim_z), vec_np, color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color='#555', linewidth=0.8)
    ax.set_xlabel("z dimension  (0 – 127)", color='#aaa', fontsize=9)
    ax.set_ylabel("component weight\n(unit-normalised)", color='#aaa', fontsize=9)
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#333')

    pos_patch = mpatches.Patch(color='#2ecc71', label='positive weight (adds attribute)')
    neg_patch  = mpatches.Patch(color='#e74c3c', label='negative weight (suppresses attribute)')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc')

    ax.set_title(
        "Semantic vector — per-dimension weights\n"
        "Dimensions with large absolute values carry the most attribute signal",
        color='white', fontsize=10
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Manipulate images using the semantic vector
# ─────────────────────────────────────────────────────────────────────────────

def manipulate_image(z_source, y, semantic_vec, model, device,
                      attribute_name="attribute",
                      alphas=None,
                      truncation=0.4):
    """
    Apply a semantic vector to an image across a range of α values.

    The manipulation formula is:
        z_new = z_source + α × semantic_vec

    α = 0        → original image (no change)
    α > 0        → attribute is added / increased
    α < 0        → attribute is removed / reversed

    α is measured in standard deviations of the z distribution (roughly),
    so sensible values are in the range [-4, +4]. Values beyond that push
    z into regions the GAN was never trained on and produce artifacts.

    Parameters
    ----------
    z_source     : torch.Tensor — shape (1, dim_z), the base image's z vector
                                  (from z_bank[index].unsqueeze(0))
    y            : torch.Tensor — shape (1, 1000), class embedding on device
    semantic_vec : torch.Tensor — shape (1, dim_z), from compute_semantic_vector()
    model        : BigGAN
    device       : str
    attribute_name : str        — label for the plot title
    alphas       : list[float]  — manipulation strengths; default is symmetric
                                  range from -3 to +3 in 9 steps
    truncation   : float        — should match the value used at generation time

    Returns
    -------
    frames  : list[PIL.Image]   — one image per α value
    """
    if alphas is None:
        alphas = np.linspace(-3, 3, 9).tolist()

    n = len(alphas)
    frames = []

    # Move semantic_vec to device only for the forward pass
    sv = semantic_vec.to(device)
    z  = z_source.to(device)

    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            z_new = z + alpha * sv
            with torch.amp.autocast(device_type="cuda"):
                out = model(z_new, y, truncation=truncation)
            frames.extend(convert_to_images(out.cpu()))

    # ── Plot: one row of images across α range
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, frame, alpha in zip(axes, frames, alphas):
        ax.imshow(frame)

        # Colour-code: blue = removing attribute, white = original, green = adding
        if abs(alpha) < 0.15:
            col = '#ffffff'
            label = f"α={alpha:.1f}\n(original)"
            for spine in ax.spines.values():
                spine.set_edgecolor('#ffffff')
                spine.set_linewidth(2)
        elif alpha > 0:
            col = '#2ecc71'
            label = f"α=+{alpha:.1f}"
        else:
            col  = '#3498db'
            label = f"α={alpha:.1f}"

        ax.set_title(label, fontsize=8, color=col, fontweight='bold', pad=3)
        ax.axis("off")

    plt.suptitle(
        f"Semantic manipulation — attribute: '{attribute_name}'\n"
        "← removing attribute                                adding attribute →",
        fontsize=11, color='white', y=1.03
    )
    plt.tight_layout()
    plt.show()

    return frames


def compare_multiple_images(z_bank, y, semantic_vec, model, device,
                              source_indices,
                              attribute_name="attribute",
                              alphas=None,
                              truncation=0.4):
    """
    Apply the same semantic vector to multiple source images simultaneously.

    This is the key test of whether your semantic vector is truly encoding
    an attribute vs. overfitting to a specific z vector. If the same
    semantic vector produces the same visual change across many different
    source images (different poses, backgrounds, lighting), the vector
    is a genuine latent direction for that attribute.

    Each row = one source image
    Each column = one α value

    Parameters
    ----------
    z_bank         : torch.Tensor  — full z_bank from generate_image_grid()
    y              : torch.Tensor  — class embedding on device
    semantic_vec   : torch.Tensor  — shape (1, dim_z)
    model          : BigGAN
    device         : str
    source_indices : list[int]     — which images from z_bank to use as sources
    attribute_name : str
    alphas         : list[float]   — default: [-2, -1, 0, +1, +2]
    truncation     : float
    """
    if alphas is None:
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]

    n_sources = len(source_indices)
    n_alphas  = len(alphas)
    sv = semantic_vec.to(device)

    # Generate all images: (n_sources × n_alphas)
    all_frames = []
    model.eval()
    with torch.no_grad():
        for src_idx in source_indices:
            row_frames = []
            z = z_bank[src_idx].unsqueeze(0).to(device)
            for alpha in alphas:
                z_new = z + alpha * sv
                with torch.amp.autocast(device_type="cuda"):
                    out = model(z_new, y, truncation=truncation)
                row_frames.extend(convert_to_images(out.cpu()))
            all_frames.append(row_frames)

    # ── Plot: one row per source image, one column per α
    fig, axes = plt.subplots(n_sources, n_alphas,
                              figsize=(2.4 * n_alphas, 2.6 * n_sources))
    fig.patch.set_facecolor('#1a1a2e')

    # Force 2D indexing even when n_sources == 1
    if n_sources == 1:
        axes = axes[np.newaxis, :]

    for row, (src_idx, row_frames) in enumerate(zip(source_indices, all_frames)):
        for col, (frame, alpha) in enumerate(zip(row_frames, alphas)):
            ax = axes[row, col]
            ax.imshow(frame)
            ax.axis("off")

            if row == 0:
                if abs(alpha) < 0.01:
                    col_label = f"α={alpha:.1f}\n(original)"
                    color = '#ffffff'
                elif alpha > 0:
                    col_label = f"α=+{alpha:.1f}"
                    color = '#2ecc71'
                else:
                    col_label = f"α={alpha:.1f}"
                    color = '#3498db'
                ax.set_title(col_label, fontsize=8, color=color,
                              fontweight='bold', pad=3)

        axes[row, 0].set_ylabel(f"src [{src_idx}]", fontsize=8,
                                 color='#aaa', rotation=90, labelpad=4)

    plt.suptitle(
        f"Semantic vector generalisation — '{attribute_name}'\n"
        "Each row is a different source image — same vector applied to all\n"
        "← suppress attribute                               add attribute →",
        fontsize=11, color='white', y=1.02
    )
    plt.tight_layout()
    plt.show()