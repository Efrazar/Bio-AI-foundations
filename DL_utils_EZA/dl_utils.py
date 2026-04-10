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
def run_epoch(model, optimizer, data_loader, loss_func, device, results,
              score_funcs, prefix="", desc=None, scaler=None):
    """
    scaler : GradScaler | None
    Pass the GradScaler created in train_simple_network_EZA.
    None = standard FP32 path (no AMP).
    NEVER create the scaler inside this function.
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    is_training = model.training

    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
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
                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        y_hat = model(inputs)
                        loss = loss_func(y_hat, labels)
                else:
                    y_hat = model(inputs)
                    loss = loss_func(y_hat, labels)

        running_loss.append(loss.item())

        if score_funcs and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat  = y_hat.detach().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    results[prefix + " loss"].append(np.mean(running_loss))
    if score_funcs:
        for name, score_func in score_funcs.items():
            try:
                results[prefix + " " + name].append(score_func(y_true, y_pred))
            except Exception:
                results[prefix + " " + name].append(float("NaN"))

    return end - start


def train_simple_network_EZA(model, loss_func, train_loader, test_loader=None,
                              score_funcs=None, epochs=300, device='cpu',
                              checkpoint_file=None, resume_from_checkpoint=None,
                              checkpoint_every_x=None, use_amp=True):
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

    Returns: pd.DataFrame with training history
    """
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    if score_funcs:
        for eval_score in score_funcs:
            to_track.append("train " + eval_score)
            if test_loader is not None:
                to_track.append("test " + eval_score)

    results = {item: [] for item in to_track}

    optimizer   = torch.optim.Adam(model.parameters(), lr=0.001)
    amp_enabled = use_amp and (device == "cuda")
    scaler      = torch.amp.GradScaler() if amp_enabled else None

    start_epoch = 0

    if resume_from_checkpoint is not None:
        checkpoint = torch.load(resume_from_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        results = checkpoint.get('results', results)
        total_train_time = results["total time"][-1] if results["total time"] else 0
        print(f"Resuming from epoch {start_epoch}, previous time: {total_train_time:.2f}s")

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        model = model.train()

        epoch_time = run_epoch(model, optimizer, train_loader, loss_func, device,
                               results, score_funcs, prefix="train",
                               desc="Training", scaler=scaler)

        results["total time"].append(epoch_time)
        results["epoch"].append(epoch)

        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device,
                          results, score_funcs, prefix="test", desc="Testing")

        # Periodic checkpoint
        if checkpoint_every_x is not None and (epoch + 1) % checkpoint_every_x == 0:
            ckpt_path = checkpoint_file if checkpoint_file else f'checkpoint_epoch_{epoch}.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, ckpt_path)
            print(f"Checkpoint saved at epoch {epoch}")

    # Final checkpoint
    if checkpoint_file is not None:
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results,
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, checkpoint_file)

    return pd.DataFrame.from_dict(results)


def moveTo(obj, device):
    """Recursively move tensors/collections to the target device."""
    if hasattr(obj, "to"):
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
