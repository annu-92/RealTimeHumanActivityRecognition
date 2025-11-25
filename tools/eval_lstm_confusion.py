# tools/eval_lstm_confusion.py
import os, json, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ---- Dataset (same as in train_lstm, simplified) ----
class KeypointSequenceDataset(Dataset):
    def __init__(self, seq_dir, kp_dir, seq_list, seq_length=32, use_xy_only=True, normalize=True, label2id=None):
        self.seq_dir = seq_dir
        self.kp_dir = kp_dir
        self.seq_length = seq_length
        self.use_xy_only = use_xy_only
        self.normalize = normalize
        self.items = []
        self.label2id = label2id
        self.id2label = None

        for s in seq_list:
            label_path = os.path.join(seq_dir, s, "label.txt")
            kp_path = os.path.join(kp_dir, f"{s}_kps.npy")
            if not os.path.exists(label_path) or not os.path.exists(kp_path):
                continue
            with open(label_path, "r", encoding="utf-8") as fh:
                lab = fh.read().strip()
            self.items.append((s, kp_path, lab))

        # if label2id not provided, build it here
        if self.label2id is None:
            labels = sorted(list({lab for _,_,lab in self.items}))
            self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}

    def __len__(self):
        return len(self.items)

    def _load_kp(self, path):
        arr = np.load(path).astype(np.float32)   # (T,33,4)
        if arr.ndim == 3:
            T, J, D = arr.shape
        elif arr.ndim == 2:
            arr = arr.reshape((-1, 33, 4))
            T, J, D = arr.shape
        else:
            raise ValueError("Unexpected keypoint shape: " + str(arr.shape))
        if self.use_xy_only:
            arr = arr[:, :, :2]   # x,y
            D = 2
        arr_flat = arr.reshape(T, J*D)
        if self.normalize:
            m = arr_flat.mean(axis=0, keepdims=True)
            s = arr_flat.std(axis=0, keepdims=True) + 1e-6
            arr_flat = (arr_flat - m) / s
        return arr_flat

    def __getitem__(self, idx):
        s, kp_path, lab = self.items[idx]
        x = self._load_kp(kp_path)
        T, F = x.shape
        if T >= self.seq_length:
            start = max(0, (T - self.seq_length)//2)
            x = x[start:start+self.seq_length, :]
        else:
            pad = np.zeros((self.seq_length - T, F), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        y = self.label2id[lab]
        return torch.from_numpy(x), torch.tensor(y), s

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    names = [b[2] for b in batch]
    return xs, ys, names

# ---- Model (same as training) ----
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=7, bidirectional=True, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        hdim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def load_splits():
    base = "data/MERL/splits"
    test = []
    with open(os.path.join(base, "test.txt"), "r", encoding="utf-8") as fh:
        test = [l.strip() for l in fh if l.strip()]
    return test

def main():
    seq_dir = "data/MERL/sequences_semantic"
    kp_dir  = "data/MERL/keypoints_semantic"

    # load label mapping from training
    mapping_path = "models/label_map_trainer.json"
    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    label2id = mapping["label2id"]
    id2label = {int(k):v for k,v in mapping["id2label"].items()} if isinstance(mapping["id2label"], dict) else mapping["id2label"]

    test_list = load_splits()
    test_ds = KeypointSequenceDataset(seq_dir, kp_dir, test_list, seq_length=32, use_xy_only=True, normalize=True, label2id=label2id)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # infer input_size from one batch
    sample_x, _, _ = next(iter(test_loader))
    input_size = sample_x.shape[-1]
    num_classes = len(test_ds.label2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model weights
    ckpt = torch.load("models/final_lstm.pt", map_location=device)
    model = LSTMPredictor(input_size=input_size, hidden_size=256, num_layers=2, num_classes=num_classes, bidirectional=True, dropout=0.4)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x,y,_ in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(y.cpu().numpy().tolist())

    # confusion matrix and report
    labels_order = [test_ds.label2id[l] for l in sorted(test_ds.label2id.keys())]
    names_order  = sorted(test_ds.label2id.keys())
    cm = confusion_matrix(all_trues, all_preds, labels=labels_order)
    print("Classification report:")
    print(classification_report(all_trues, all_preds, target_names=names_order, digits=4))

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix (Test)")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(names_order)))
    ax.set_yticks(range(len(names_order)))
    ax.set_xticklabels(names_order, rotation=45, ha="right")
    ax.set_yticklabels(names_order)
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    out_path = "models/confusion_matrix_test.png"
    plt.savefig(out_path, dpi=200)
    print("Saved confusion matrix image to", out_path)

if __name__ == "__main__":
    main()
