import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from tqdm.autonotebook import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
import os
import time
from glob import glob
from skimage.io import imread

# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 constants
# ─────────────────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_MEAN = [0.49139968, 0.48215841, 0.44653091]
CIFAR10_STD  = [0.24703223, 0.24348513, 0.26158784]

CIFAR10_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])


# ─────────────────────────────────────────────────────────────────────────────
# Load_model_EZA
# ─────────────────────────────────────────────────────────────────────────────
def load_model_EZA(model, checkpoint_path, device='cpu'):
    """
    Load a saved model checkpoint back into a model instance.

    Parameters
    ----------
    model           : nn.Module — instantiated model with SAME architecture as training
    checkpoint_path : str       — path to the .tar checkpoint file
    device          : str       — 'cuda' or 'cpu'

    Returns
    -------
    model      : nn.Module — model with loaded weights, set to eval() mode
    checkpoint : dict      — full checkpoint dict (epoch, results, etc.)

    Usage
    -----
    model, ckpt = load_model_EZA(cifar10_cnn_model_pooled,
                                 "./checkpoints/cnn_base_final.tar",
                                 device=device)
    print(f"Loaded from epoch {ckpt['epoch']}")
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()   # CRITICAL: disables BatchNorm/Dropout training behaviour

    epoch_loaded = checkpoint.get('epoch', 'unknown')
    print(f"Model loaded from '{checkpoint_path}'")
    print(f"  Checkpoint epoch : {epoch_loaded}")
    print(f"  Device           : {device}")
    print(f"  Mode             : eval()")

    return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Predict_single_image_EZA
# ─────────────────────────────────────────────────────────────────────────────
def predict_single_image_EZA(image_input, model, device,
                              classes=None,
                              transform=None,
                              show_plot=True):
    """
    Run inference on a single image and return predicted class + probabilities.

    Parameters
    ----------
    image_input : str | PIL.Image — file path OR a PIL Image object
    model       : nn.Module       — trained model already in eval() mode
    device      : str             — 'cuda' or 'cpu'
    classes     : list[str]       — class names (default: CIFAR-10 labels)
    transform   : transforms.*    — preprocessing pipeline (default: CIFAR-10)
    show_plot   : bool            — display the image + probability bar chart

    Returns
    -------
    pred_class  : str        — predicted class name
    confidence  : float      — confidence as a percentage (0-100)
    probs       : np.ndarray — full probability vector (len = n_classes)

    Usage
    -----
    pred, conf, probs = predict_single_image_EZA("my_photo.jpg", model, device=device)
    """
    if classes is None:
        classes = CIFAR10_CLASSES
    if transform is None:
        transform = CIFAR10_INFERENCE_TRANSFORM

    # Load image
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    else:
        img_pil = image_input.convert("RGB")

    # Preprocess: apply transforms + add batch dimension [1, 3, 32, 32]
    tensor = transform(img_pil).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    pred_class = classes[pred_idx.item()]
    confidence = conf.item() * 100
    probs_np   = probs.squeeze().cpu().numpy()

    print(f"Predicted : {pred_class.upper()}")
    print(f"Confidence: {confidence:.1f}%")

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left panel: original image (upsampled for visibility — CIFAR is tiny 32x32)
        axes[0].imshow(img_pil.resize((128, 128), Image.NEAREST))
        axes[0].set_title(f"Predicted: {pred_class}\nConfidence: {confidence:.1f}%",
                          fontsize=12, fontweight='bold')
        axes[0].axis("off")

        # Right panel: horizontal probability bar chart
        colors = ['#2ecc71' if c == pred_class else '#3498db' for c in classes]
        axes[1].barh(classes, probs_np * 100, color=colors)
        axes[1].set_xlabel("Confidence (%)")
        axes[1].set_title("Class Probabilities")
        axes[1].set_xlim(0, 100)
        for i, v in enumerate(probs_np * 100):
            axes[1].text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=8)

        green_patch = mpatches.Patch(color='#2ecc71', label='Top prediction')
        blue_patch  = mpatches.Patch(color='#3498db', label='Other classes')
        axes[1].legend(handles=[green_patch, blue_patch], fontsize=8)

        plt.suptitle("CIFAR-10 Inference — Single Image", fontsize=13)
        plt.tight_layout()
        plt.show()

    return pred_class, confidence, probs_np


# ─────────────────────────────────────────────────────────────────────────────
# Predict_batch_EZA
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch_EZA(dataset, model, device, n=12,
                      classes=None, transform=None, random_seed=42):
    """
    Run inference on a random batch of images from a dataset and display results.

    Parameters
    ----------
    dataset     : torchvision Dataset — e.g. test_raw (raw PIL images, no transforms)
    model       : nn.Module           — trained model already in eval() mode
    device      : str                 — 'cuda' or 'cpu'
    n           : int                 — number of images to sample (default 12)
    classes     : list[str]           — class names (default: CIFAR-10)
    transform   : transforms.*        — preprocessing pipeline (default: CIFAR-10)
    random_seed : int                 — for reproducibility

    Returns
    -------
    results_df  : pd.DataFrame — columns: index, true_label, pred_label,
                                          confidence, correct

    Usage
    -----
    df = predict_batch_EZA(test_raw, model, device=device, n=12)
    print(df['correct'].mean())   # quick batch accuracy
    """
    if classes is None:
        classes = CIFAR10_CLASSES
    if transform is None:
        transform = CIFAR10_INFERENCE_TRANSFORM

    np.random.seed(random_seed)
    indices = np.random.choice(len(dataset), n, replace=False)

    records = []
    cols    = int(np.ceil(np.sqrt(n)))
    rows    = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).flatten()

    model.eval()
    for i, idx in enumerate(indices):
        img_pil, true_label = dataset[idx]

        tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_class = classes[pred_idx.item()]
        true_class = classes[true_label]
        confidence = conf.item() * 100
        is_correct = (pred_class == true_class)

        records.append({
            'index'     : int(idx),
            'true_label': true_class,
            'pred_label': pred_class,
            'confidence': round(confidence, 2),
            'correct'   : is_correct
        })

        axes[i].imshow(img_pil.resize((64, 64), Image.NEAREST))
        color = "#27ae60" if is_correct else "#e74c3c"
        axes[i].set_title(
            f"True : {true_class}\nPred : {pred_class}\n({confidence:.0f}%)",
            fontsize=8, color=color, fontweight='bold'
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    results_df = pd.DataFrame(records)
    batch_acc  = results_df['correct'].mean() * 100

    plt.suptitle(
        f"CIFAR-10 Batch Inference  |  n={n}  |  Batch Accuracy: {batch_acc:.1f}%\n"
        "Green = Correct   Red = Incorrect",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

    print(f"\nBatch accuracy : {batch_acc:.1f}%  ({results_df['correct'].sum()}/{n} correct)")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Training Model Functions
# ─────────────────────────────────────────────────────────────────────────────
def _compact_predictions(y_hat, labels, ignore_index=None):
    """
    Shrink model outputs ON THE GPU before they are copied to the CPU.

    Why: the old code copied full logit tensors to the CPU and kept them in
    Python lists. For CIFAR-10 that is 10 floats per image (fine). For a
    Seq2Seq model with a 24,577-word vocabulary it is
    ~60k sentences x ~7 steps x 24,577 floats ≈ 10 billion numbers per epoch,
    far beyond 31 GB of RAM. Taking argmax on the GPU first sends 1 integer
    per prediction across the Thunderbolt link instead of V floats.

    Handles:
      - tuple outputs (e.g. (logits, attention) from a model in eval mode)
      - (B, C) classification logits     -> (B,) class ids
      - (B, T, V) sequence logits         -> flattened token ids, PAD removed
      - (B, 1) / (B,) regression outputs  -> kept as values
    """
    if isinstance(y_hat, tuple):
        y_hat = y_hat[0]
    y_hat = y_hat.detach()

    if y_hat.dim() == 3 and labels.dim() == 2:           # sequence model
        T = min(y_hat.size(1), labels.size(1))
        preds = y_hat[:, :T].argmax(dim=-1).reshape(-1)
        labels = labels[:, :T].reshape(-1)
        if ignore_index is not None:
            keep = labels != ignore_index
            preds, labels = preds[keep], labels[keep]
    elif y_hat.dim() == 2 and y_hat.size(1) > 1:         # classification
        preds = y_hat.argmax(dim=1)
    else:                                                # regression
        preds = y_hat.float()

    return labels.cpu().numpy(), preds.cpu().numpy()


def run_epoch(model, optimizer, data_loader, loss_func, device, results,
              score_funcs, prefix="", desc=None, scaler=None, ignore_index=None):
    """
    scaler : GradScaler | None
        Pass the GradScaler created in train_*_network_EZA.
        None = standard FP32 path (no AMP).  NEVER create the scaler in here.
        In eval mode the scaler is only used as an "AMP on" flag (autocast).
    ignore_index : int | None
        Label value to drop before scoring (e.g. PAD=0 for sequence models).

    Memory notes (RTX 2080 Ti on a Thunderbolt 3 eGPU):
      - Loss is accumulated ON the GPU and read once per epoch.  Calling
        loss.item() every batch forces the CPU to wait for the GPU
        (a synchronization point) on every step.
      - Predictions are reduced with argmax on the GPU before copying.
      - Arrays are concatenated once at the end, not via Python lists of floats.
    """
    running_loss = torch.zeros((), device=device)
    n_batches = 0
    y_true, y_pred = [], []
    start = time.time()
    is_training = model.training
    use_autocast = scaler is not None

    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.amp.autocast(device_type="cuda"):
                    y_hat = model(inputs)
                    loss = loss_func(y_hat, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_hat = model(inputs)
                loss = loss_func(y_hat, labels)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast(device_type="cuda"):
                        y_hat = model(inputs)
                        loss = loss_func(y_hat, labels)
                else:
                    y_hat = model(inputs)
                    loss = loss_func(y_hat, labels)

        running_loss += loss.detach().float()
        n_batches += 1

        if score_funcs and isinstance(labels, torch.Tensor):
            t, p = _compact_predictions(y_hat, labels, ignore_index)
            y_true.append(t)
            y_pred.append(p)

    end = time.time()

    results[prefix + " loss"].append((running_loss / max(n_batches, 1)).item())
    if score_funcs:
        y_true = np.concatenate(y_true) if y_true else np.array([])
        y_pred = np.concatenate(y_pred) if y_pred else np.array([])
        for name, score_func in score_funcs.items():
            try:
                results[prefix + " " + name].append(score_func(y_true, y_pred))
            except Exception:
                results[prefix + " " + name].append(float("NaN"))

    return end - start


def train_simple_network_EZA(model, loss_func, train_loader, test_loader=None,
                              score_funcs=None, epochs=300, device='cpu',
                              checkpoint_file=None, resume_from_checkpoint=None,
                              checkpoint_every_x=None, use_amp=True, ignore_index=None):
    """
    Train simple neural networks with EZA improvements.

    Keyword arguments:
    model                  -- PyTorch model to train
    loss_func              -- loss function (outputs, labels) -> score
    train_loader           -- DataLoader returning (input, label) tuples
    test_loader            -- Optional DataLoader for evaluation after every epoch
    score_funcs            -- dict of scoring functions e.g. {'Accuracy': accuracy_score}
    epochs                 -- number of training epochs
    device                 -- 'cuda' or 'cpu'
    checkpoint_file        -- path to save final checkpoint
    resume_from_checkpoint -- path to load checkpoint from to resume training
    checkpoint_every_x     -- save intermediate checkpoint every X epochs
    use_amp                -- enable Automatic Mixed Precision (RTX GPUs)
    ignore_index           -- label value dropped before scoring (e.g. PAD token)

    Returns: pd.DataFrame with training history
    """
    return train_network_EZA(model, loss_func, train_loader, test_loader=test_loader,
                             score_funcs=score_funcs, epochs=epochs, device=device,
                             checkpoint_file=checkpoint_file,
                             resume_from_checkpoint=resume_from_checkpoint,
                             checkpoint_every_x=checkpoint_every_x, use_amp=use_amp,
                             optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                             ignore_index=ignore_index)


def moveTo(obj, device):
    """
    Recursively move tensors/collections to the target device.
    non_blocking=True lets the copy overlap with GPU compute when the
    DataLoader uses pin_memory=True — useful over a Thunderbolt link.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        return {moveTo(k, device): moveTo(v, device) for k, v in obj.items()}
    else:
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Improved Neural Network EZA
# ─────────────────────────────────────────────────────────────────────────────
def train_network_EZA(model,
                      loss_func,
                      train_loader,
                      test_loader=None,
                      score_funcs=None,
                      epochs=300,
                      device='cpu',
                      checkpoint_file=None,
                      resume_from_checkpoint=None,
                      checkpoint_every_x=None,
                      use_amp=True,
                      optimizer=None,
                      lr_schedule=None,
                      val_loader=None,
                      ignore_index=None):
    """
    Train neural networks with EZA improvements.

    Keyword arguments:
    model                  -- PyTorch model to train
    loss_func              -- loss function (outputs, labels) -> score
    train_loader           -- DataLoader returning (input, label) tuples
    test_loader            -- Optional DataLoader for evaluation after every epoch
    score_funcs            -- dict of scoring functions e.g. {'Accuracy': accuracy_score}
    epochs                 -- number of training epochs
    device                 -- 'cuda' or 'cpu'
    checkpoint_file        -- path to save final checkpoint
    resume_from_checkpoint -- path to load checkpoint from to resume training
    checkpoint_every_x     -- save intermediate checkpoint every X epochs
    use_amp                -- enable Automatic Mixed Precision (RTX GPUs)
    optimizer              -- if None, AdamW with default settings
    lr_schedule            -- any torch LR scheduler. ReduceLROnPlateau monitors
                              val loss if val_loader is given, else test loss.
    val_loader             -- Optional DataLoader to evaluate on after every epoch
    ignore_index           -- label value dropped before scoring (e.g. PAD token)

    Returns: pd.DataFrame with training history
    """
    device = str(device)
    to_track = ["epoch", "total time", "train loss"]
    if val_loader is not None:
        to_track.append("val loss")
    if test_loader is not None:
        to_track.append("test loss")
    if score_funcs:
        for eval_score in score_funcs:
            to_track.append("train " + eval_score)
            if val_loader is not None:
                to_track.append("val " + eval_score)
            if test_loader is not None:
                to_track.append("test " + eval_score)

    results = {item: [] for item in to_track}

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())
    amp_enabled = use_amp and device.startswith("cuda")
    scaler      = torch.amp.GradScaler() if amp_enabled else None

    start_epoch = 0
    epoch = start_epoch - 1

    if resume_from_checkpoint is not None:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint['epoch'] + 1
        epoch = checkpoint['epoch']
        results = checkpoint.get('results', results)
        total_train_time = sum(results["total time"]) if results["total time"] else 0
        print(f"Resuming from epoch {start_epoch}, previous time: {total_train_time:.2f}s")

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    def _save(path, epoch):
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results,
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, path)

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        model = model.train()

        epoch_time = run_epoch(model, optimizer, train_loader, loss_func, device,
                               results, score_funcs, prefix="train",
                               desc="Training", scaler=scaler, ignore_index=ignore_index)

        results["total time"].append(epoch_time)
        results["epoch"].append(epoch)

        # Evaluation also runs under autocast when AMP is on: ~half the
        # activation memory and faster on Turing Tensor Cores.
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device,
                          results, score_funcs, prefix="test", desc="Testing",
                          scaler=scaler, ignore_index=ignore_index)

        if val_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, val_loader, loss_func, device,
                          results, score_funcs, prefix="val", desc="Validating",
                          scaler=scaler, ignore_index=ignore_index)

        # In PyTorch, the convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor = "val loss" if val_loader is not None else "test loss"
                if monitor not in results:
                    raise ValueError("ReduceLROnPlateau needs a val_loader or test_loader.")
                lr_schedule.step(results[monitor][-1])
            else:
                lr_schedule.step()

        # Periodic checkpoint
        if checkpoint_every_x is not None and (epoch + 1) % checkpoint_every_x == 0:
            ckpt_path = checkpoint_file if checkpoint_file else f'checkpoint_epoch_{epoch}.tar'
            _save(ckpt_path, epoch)
            print(f"Checkpoint saved at epoch {epoch}")

    # Final checkpoint
    if checkpoint_file is not None:
        _save(checkpoint_file, epoch)

    return pd.DataFrame.from_dict(results)


# ─────────────────────────────────────────────────────────────────────────────
# DSB2018 segmentation dataset
# ─────────────────────────────────────────────────────────────────────────────
class DSB2018_EZA(Dataset):
    """
    Dataset class for the 2018 Data Science Bowl (DSB2018) — Kaggle.

    Loads nucleus microscopy images and their corresponding binary segmentation masks.

    Optimized for:
        - RTX 2080 Ti (10.57 GiB VRAM, Turing sm_75)
        - FP16 / AMP training (half-precision tensors on GPU)
        - PyTorch 2.9.1 + CUDA 12.8

    Args:
        paths  (list[str]) : List of sample root directories. Each directory must contain:
                               <path>/images/<image>.png   — one RGB image
                               <path>/masks/<mask_n>.png   — one or more binary masks
        size   (int)       : Target spatial resolution for both image and mask (default 256).
        device (str)       : kept for backwards compatibility; tensors are returned
                             on the CPU so DataLoader workers + pin_memory can handle them.
    """

    def __init__(self, paths: list, size: int = 256, device: str = "cpu"):
        self.paths  = paths
        self.size   = size
        self.device = device

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # ── 1. Resolve file paths ──────────────────────────────────────────────
        img_path   = glob(self.paths[idx] + "/images/*")[0]
        mask_paths = glob(self.paths[idx] + "/masks/*")   # N individual nucleus masks

        # ── 2. Load image ──────────────────────────────────────────────────────
        img = imread(img_path)[:, :, :3]          # (H, W, 3)  uint8
        # float32 is intentional: autocast casts to FP16 inside the model.
        img = np.moveaxis(img, -1, 0).astype(np.float32) / 255.0  # (3, H, W)

        # ── 3. Load & merge binary masks ───────────────────────────────────────
        # Merge in-place with a bool buffer: no float64 copies of every mask.
        final_mask = None
        for f in mask_paths:
            m = imread(f) > 0
            final_mask = m if final_mask is None else (final_mask | m)
        final_mask = final_mask.astype(np.float32)             # (H, W)

        # ── 4. NumPy → PyTorch tensors ─────────────────────────────────────────
        img_t  = torch.from_numpy(img)                         # (3, H, W)
        mask_t = torch.from_numpy(final_mask).unsqueeze(0)     # (1, H, W)

        # ── 5. Spatial resize to (size × size) ────────────────────────────────
        img_t = F.interpolate(img_t.unsqueeze(0), size=(self.size, self.size),
                              mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=(self.size, self.size),
                               mode="nearest").squeeze(0)

        return img_t, mask_t   # both float32; AMP handles FP16 casting during training
