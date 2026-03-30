import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import time

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
            # ✅ zero_grad BEFORE forward pass, with set_to_none for speed
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                # ── AMP path ────────────────────────────────────────────────
                with torch.amp.autocast(device_type="cuda"):   # ← FP16 zone
                    y_hat = model(inputs)
                    loss  = loss_func(y_hat, labels)

                scaler.scale(loss).backward()   # scale before backprop
                scaler.step(optimizer)           # unscale + step
                scaler.update()                  # adjust scale factor

            else:
                # ── FP32 path ────────────────────────────────────────────────
                y_hat = model(inputs)
                loss  = loss_func(y_hat, labels)
                loss.backward()
                optimizer.step()

        else:
            # ── Evaluation path (no gradients) ───────────────────────────────
            with torch.no_grad():
                if scaler is not None:
                    with torch.amp.autocast(device_type="cuda"):
                        y_hat = model(inputs)
                        loss  = loss_func(y_hat, labels)
                else:
                    y_hat = model(inputs)
                    loss  = loss_func(y_hat, labels)

        running_loss.append(loss.item())

        if score_funcs and isinstance(labels, torch.Tensor):  # ✅ null guard
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

def train_simple_network_EZA(model, loss_func, train_loader, test_loader=None, score_funcs=None, 
                                epochs=300, device='cpu', checkpoint_file=None, resume_from_checkpoint=None,
                                checkpoint_every_x=None, use_amp=True):
    
    """ Train simple neural networks with EZA improvements based on my hardware configuration
    
    Keyword arguments:
    model -- the PyTorch model / "Module" to train
    loss_func -- the loss function that takes in batch in two arguments, the model outputs and the labels, and returns a score
    train_loader -- PyTorch DataLoader object that returns tuples of (input, label) pairs. 
    test_loader -- Optional PyTorch DataLoader to evaluate on after every epoch
    score_funcs -- A dictionary of scoring functions to use to evalue the performance of the model
    epochs -- the number of training epochs to perform
    device -- the compute lodation to perform training
    checkpoint_file -- path to save checkpoint at the end of training
    resume_from_checkpoint -- path to load checkpoint from to resume training
    checkpoint_every_x -- save checkpoint every X epochs (None to only save at end)
    
    """

    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    if score_funcs:
        for eval_score in score_funcs:
            to_track.append("train " + eval_score )
            if test_loader is not None:
                to_track.append("test " + eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []
    
    # Better for CNNs training with Adam.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    amp_enabled = use_amp and (device == "cuda")
    scaler = GradScaler() if amp_enabled else None # Created once
    
    # Initialize starting epoch
    start_epoch = 0
    
    # Load checkpoint if provided
    if resume_from_checkpoint is not None:
        checkpoint = torch.load(resume_from_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        results = checkpoint.get('results', results)  # Restore previous results
        total_train_time = results["total time"][-1] if results["total time"] else 0
        print(f"Resuming from epoch {start_epoch}, previous training time: {total_train_time:.2f}s")
    
    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    
    # Move optimizer state to device (important when loading on GPU)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch"):
        model = model.train()#Put our model in training mode
        
        epoch_time = run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training", scaler=scaler)

        results["total time"].append(epoch_time)
        results["epoch"].append(epoch)
        
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
        
        # Save checkpoint every X epochs if specified
        if checkpoint_every_x is not None and (epoch + 1) % checkpoint_every_x == 0:
            checkpoint_path = checkpoint_file if checkpoint_file else f'checkpoint_epoch_{epoch}.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results' : results
                }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
                    
    # Save final checkpoint if specified
    if checkpoint_file is not None:
        payload = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results":              results,
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()  # ← preserve scale factor
        torch.save(payload, path)
                
    return pd.DataFrame.from_dict(results)

def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj