import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd

# --------- Dataset ---------
class KeypointSequenceDataset(Dataset):
    def __init__(self, seq_dir, kp_dir, split_list=None, seq_length=32, use_xy_only=True, normalize=True):
        self.seq_dir = seq_dir
        self.kp_dir = kp_dir
        self.seq_length = seq_length
        self.use_xy_only = use_xy_only
        self.normalize = normalize


        seq_folders = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir,d))])
        if split_list is not None:
            seq_folders = [s for s in seq_folders if s in set(split_list)]
        self.items = []
        for s in seq_folders:
            label_path = os.path.join(seq_dir, s, "label.txt")
            kp_path = None
            # try both naming conventions for keypoints file
            candidate1 = os.path.join(kp_dir, f"{s}_kps.npy")
            candidate2 = os.path.join(kp_dir, f"{s}.npy")
            if os.path.exists(candidate1):
                kp_path = candidate1
            elif os.path.exists(candidate2):
                kp_path = candidate2
            else:
                g = glob.glob(os.path.join(kp_dir, f"{s}*"))
                if g:
                    kp_path = g[0]
            if kp_path is None:
                continue
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r", encoding="utf-8") as fh:
                label = fh.read().strip()
            self.items.append((s, kp_path, label))
        # build label->id mapping
        labels = sorted(list({lab for (_,_,lab) in self.items}))
        self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}

    def __len__(self):
        return len(self.items)

    def _load_kp(self, path):
        arr = np.load(path)   
        arr = arr.astype(np.float32)

        if arr.ndim == 3:
            T, J, D = arr.shape
        elif arr.ndim == 2:
            arr = arr.reshape((-1, 33, 4))
            T, J, D = arr.shape
        else:
            raise ValueError("Unexpected kp array shape: "+str(arr.shape))
        
        if self.use_xy_only:
            arr = arr[:,:, :2]   
            D = 2
       
        arr_flat = arr.reshape(T, J * D)
      
        if self.normalize:
            mean = arr_flat.mean(axis=0, keepdims=True)
            std = arr_flat.std(axis=0, keepdims=True) + 1e-6
            arr_flat = (arr_flat - mean) / std
        return arr_flat  # (T, feat)

    def __getitem__(self, idx):
        s, kp_path, label = self.items[idx]
        x = self._load_kp(kp_path)  # (T, F)
        T, F = x.shape
        # pad or trim to seq_length
        if T >= self.seq_length:
            # center-crop or take first seq_length frames (here take center)
            start = max(0, (T - self.seq_length)//2)
            x = x[start: start + self.seq_length, :]
        else:
            # pad with zeros
            pad = np.zeros((self.seq_length - T, F), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        y = self.label2id[label]
        # return tensor
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), s

# --------- Model ---------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=10, bidirectional=True, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        lstm_output_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, T, feat)
        out, (hn, cn) = self.lstm(x)  # out: (batch, T, hidden*directions)
        # take last timestep output
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

# --------- Utilities ---------
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    seq_names = [b[2] for b in batch]
    return xs, ys, seq_names

def train_epoch(model, device, loader, opt, criterion):
    model.train()
    losses = []
    preds_all = []
    trues_all = []
    for x,y,_ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_all.extend(preds.tolist())
        trues_all.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues_all, preds_all)
    return np.mean(losses), acc

def eval_epoch(model, device, loader, criterion, id2label):
    model.eval()
    losses = []
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for x,y,_ in tqdm(loader, desc="eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            trues_all.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues_all, preds_all)
    f1 = f1_score(trues_all, preds_all, average='macro')
    cm = confusion_matrix(trues_all, preds_all)
    # pretty class report
    report = classification_report(trues_all, preds_all, target_names=[id2label[i] for i in sorted(id2label.keys())], zero_division=0)
    return np.mean(losses), acc, f1, cm, report

# --------- Main ---------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # find sequences
    print("Loading dataset lists...")
    # read splits if exist, otherwise split by sequences list
    split_files = {}
    for sp in ["train","val","test"]:
        path = os.path.join("data","MERL","splits", f"{sp}.txt")
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as fh:
                split_files[sp] = [l.strip() for l in fh if l.strip()]
    # if splits provided, use them, else split by ratio
    if split_files:
        train_list = split_files.get("train", [])
        val_list = split_files.get("val", [])
        test_list = split_files.get("test", [])
    else:
        seq_folders = sorted([d for d in os.listdir(args.seq_dir) if os.path.isdir(os.path.join(args.seq_dir,d))])
        random.shuffle(seq_folders)
        n = len(seq_folders)
        train_list = seq_folders[:int(0.7*n)]
        val_list = seq_folders[int(0.7*n):int(0.9*n)]
        test_list = seq_folders[int(0.9*n):]

    print("Num sequences: train", len(train_list), "val", len(val_list), "test", len(test_list))

    train_ds = KeypointSequenceDataset(args.seq_dir, args.kp_dir, split_list=train_list, seq_length=args.seq_len, use_xy_only=not args.use_xyz, normalize=not args.no_normalize)
    val_ds = KeypointSequenceDataset(args.seq_dir, args.kp_dir, split_list=val_list, seq_length=args.seq_len, use_xy_only=not args.use_xyz, normalize=not args.no_normalize)
    test_ds = KeypointSequenceDataset(args.seq_dir, args.kp_dir, split_list=test_list, seq_length=args.seq_len, use_xy_only=not args.use_xyz, normalize=not args.no_normalize)

    print("Label set:", train_ds.label2id)
    n_classes = len(train_ds.label2id)
    # freeze label mapping file for later
    with open("models/label_map_trainer.json","w",encoding="utf-8") as fh:
        json.dump({"label2id": train_ds.label2id, "id2label": train_ds.id2label}, fh, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    sample_x, _, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    print("Input size:", input_size, "seq_len:", sample_x.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    model = LSTMPredictor(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=n_classes, bidirectional=not args.unidirectional, dropout=args.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_val = -1.0
    os.makedirs("models", exist_ok=True)
    for epoch in range(1, args.epochs+1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(model, device, train_loader, opt, criterion)
        val_loss, val_acc, val_f1, val_cm, val_report = eval_epoch(model, device, val_loader, criterion, train_ds.id2label)
        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")
        print(val_report)
        scheduler.step(val_acc)

        # save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "label2id": train_ds.label2id,
                "id2label": train_ds.id2label,
            }
            torch.save(ckpt, os.path.join("models", f"best_lstm_epoch{epoch}.pt"))
            print("Saved best model.")

    # final test eval
    test_loss, test_acc, test_f1, test_cm, test_report = eval_epoch(model, device, test_loader, criterion, train_ds.id2label)
    print("\n=== Final Test Results ===")
    print(f"Test loss: {test_loss:.4f} acc: {test_acc:.4f} f1: {test_f1:.4f}")
    print(test_report)
    # save final model
    torch.save({"model_state": model.state_dict(), "label2id": train_ds.label2id}, os.path.join("models", "final_lstm.pt"))
    print("Saved final model to models/final_lstm.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp_dir", type=str, default="data/MERL/keypoints_semantic", help="folder with keypoint .npy files")
    parser.add_argument("--seq_dir", type=str, default="data/MERL/sequences_semantic", help="folder with sequence folders containing label.txt")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--use_xyz", action="store_true", help="use x,y,z coords instead of x,y only")
    parser.add_argument("--no_normalize", action="store_true", help="disable per-sequence normalization")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
