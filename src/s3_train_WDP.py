import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple
from scripts.get_info_from_table import judge_intervals
import glob
import argparse

# ----------------------------
# 1. arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Saprot Protein Sequence Embedding from foldseek sequences.")
    parser.add_argument("--part_names", type=str, required=True, help="part names of different csv-files, seperated by comas.")
    parser.add_argument("--label_dir", type=str, default="./data/labels", 
                        help="0/1 labels of every single residue files dir of all proteins. All files should be named name_domain.txt")
    parser.add_argument("--embedding_dir", type=str, default="./output/embeddings", help="embedding dir.")
    parser.add_argument("--output_dir", type=str, default="./output/protein_domain_pred", help="model and results output dir.")
    parser.add_argument("--hidden_size", type=int, default=1280, help="SaProt_650M hidden size.")
    parser.add_argument("--lstm_hidden", type=int, default=256, help="prediction LSTM head hidden size.")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout.")
    parser.add_argument("--batch_size", type=int, default=1, help="training and validating batch size.")
    parser.add_argument("--epochs", type=int, default=51, help="training epochs.")
    parser.add_argument("--learing_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    return parser.parse_args() 

# ----------------------------
# 2. load labels
# ----------------------------
def load_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"Label file {label_path} has less than 2 lines")
        label_str = lines[1].strip()
    return np.array([int(c) for c in label_str], dtype=np.int64)

# ----------------------------
# 3. model class
# ----------------------------
class TokenClassifier(nn.Module):
    def __init__(self, hidden_size, lstm_hidden=256, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            hidden_size, lstm_hidden, 
            num_layers=1, bidirectional=True, batch_first=True
        )
        classifier_input = lstm_hidden * 2
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):  # x: [B, L, D]
        x, _ = self.lstm(x)  # [B, L, 2*H], no need to unsqueeze!
        logits = self.classifier(x).squeeze(-1)  # [B, L]
        return logits

# ----------------------------
# 4. dataset
# ----------------------------
class ProteinTokenDataset(Dataset):
    def __init__(self, seq_names, embeddings_dict, labels_dict):
        self.names = seq_names
        self.embeddings = embeddings_dict
        self.labels = labels_dict
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        emb = torch.from_numpy(self.embeddings[name]).float()
        label = torch.from_numpy(self.labels[name]).long()
        return name, emb, label

# ----------------------------
# 5. train & validate
# ----------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_tokens = 0
    for name, emb, label in dataloader:
        emb = emb.to(device)      # [L, D]
        label = label.to(device)  # [L]
        
        optimizer.zero_grad()
        logits = model(emb)       # [L]
        loss = criterion(logits, label.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(label)
        n_tokens += len(label)
    return total_loss / n_tokens

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for name, emb, label in dataloader:
            emb = emb.to(device)
            logits = model(emb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(label.numpy())
    
    all_labels_flat = np.concatenate(all_labels)
    all_preds_flat = np.concatenate(all_preds)
    
    acc = accuracy_score(all_labels_flat, all_preds_flat)
    prec = precision_score(all_labels_flat, all_preds_flat, zero_division=0)
    rec = recall_score(all_labels_flat, all_preds_flat, zero_division=0)
    f1 = f1_score(all_labels_flat, all_preds_flat, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def apply_rules_and_extract_domains(
    preds: np.ndarray,
    labels: np.ndarray,
    min_length: int = 200,
    noise_threshold: int = 3
) -> dict:
    """
    Apply post-processing rules and extract domain segments.

    Args:
        preds (np.ndarray): Predicted binary labels, shape (L,)
        labels (np.ndarray): True binary labels, shape (L,)
        min_length (int): Threshold for rule 1 (default: 200)
        noise_threshold (int): Max length of "noise" segment to flip (default: 3)

    Returns:
        dict with keys:
            - 'preds_rule': processed predictions
            - 'pred_domains': list of [start, end] for predicted 1-segments
            - 'true_domains': list of [start, end] for true 1-segments
    """
    L = len(preds)
    
    # Rule 1: short sequences → all 0
    if L < min_length:
        preds_rule = np.zeros_like(preds, dtype=int)
    else:
        preds_rule = preds.copy()
        
        # Rule 2: smooth isolated flips
        # Step 1: find all contiguous segments
        def get_segments(arr):
            segments = []
            start = 0
            current = arr[0]
            for i in range(1, len(arr)):
                if arr[i] != current:
                    segments.append((start, i - 1, current))
                    start = i
                    current = arr[i]
            segments.append((start, len(arr) - 1, current))
            return segments
        
        segments = get_segments(preds_rule)
        if len(segments) <= 1:
            # All same, nothing to smooth
            pass
        else:
            # Check internal segments (not first or last)
            for i in range(1, len(segments) - 1):
                seg_start, seg_end, seg_val = segments[i]
                seg_len = seg_end - seg_start + 1
                # If this segment is short and surrounded by opposite value
                if seg_len <= noise_threshold:
                    # Flip it to match neighbors
                    preds_rule[seg_start:seg_end + 1] = 1 - seg_val
    
    # Helper: extract continuous 1-segments
    def extract_ones_segments(arr, min_final_length=200, alpha: float = 1.0):
        """
            min_final_length: minimum of valid_domain_length
            alpha: scaling factor for gap tolerance
        """
        segments = []
        in_domain = False
        start = -1
        for i, val in enumerate(arr):
            if val == 1:
                if not in_domain:
                    start = i
                    in_domain = True
            else:
                if in_domain:
                    segments.append([start, i - 1])
                    in_domain = False
        if in_domain:
            segments.append([start, len(arr) - 1])
        if segments:
            chains = []  # list of lists of intervals
            current_chain = [segments[0]]
            
            for i in range(1, len(segments)):
                prev_s, prev_e = current_chain[-1]
                curr_s, curr_e = segments[i]
                
                gap = curr_s - prev_e - 1
                prev_len = prev_e - prev_s + 1
                curr_len = curr_e - curr_s + 1
                
                # Adaptive condition: gap < alpha * max(prev_len, curr_len)
                if gap < alpha * max(prev_len, curr_len):
                    current_chain.append(segments[i])
                else:
                    chains.append(current_chain)
                    current_chain = [segments[i]]
            
            chains.append(current_chain)  # don't forget last chain
            
            # Merge each chain into [min_start, max_end]
            merged_candidates = []
            for chain in chains:
                start = chain[0][0]
                end = chain[-1][-1]
                length = end - start + 1
                if length >= min_final_length:
                    merged_candidates.append([start, end])
            
            if not merged_candidates:
                return []
            
            # Select the candidate with maximum length
            best = max(merged_candidates, key=lambda x: x[1] - x[0] + 1)
            return best
        else:
            return []

    pred_domains = extract_ones_segments(preds_rule)
    true_domains = extract_ones_segments(labels)

    return {
        "preds_rule": preds_rule,
        "pred_domains": pred_domains,  # list of [start, end]
        "true_domains": true_domains   # list of [start, end]
    }

def is_prediction_valid(pred_domains, true_domains, valid_thr=5):
    return pred_domains and abs(pred_domains[0]-true_domains[0]) <= valid_thr and abs(pred_domains[-1]-true_domains[-1]) <= valid_thr


def evaluate_per_sequence_and_save(model, dataloader, device, output_csv):
    model.eval()
    results = []
    
    with torch.no_grad():
        for name, emb, label in tqdm(dataloader, desc="Testing per sequence"):
            emb = emb.to(device)
            label_np = label.numpy() 
            
            logits = model(emb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            label_flat = label_np.flatten()
            preds_flat = preds.flatten()
        
            res_rules = apply_rules_and_extract_domains(preds_flat, label_flat)
            preds_flat = res_rules["preds_rule"]
            pred_domains = res_rules["pred_domains"]
            true_domains = res_rules["true_domains"]
            true_label = 0
            pred_label = 0
            valid_thr = 5
            if true_domains:
                true_label = 1
                if is_prediction_valid(pred_domains, true_domains, valid_thr):
                    pred_label = 1
            else:
                true_label = 0
                pred_label = 1 if pred_domains else 0
            
            label_str = ''.join(map(str, label_flat.tolist()))
            pred_str = ''.join(map(str, preds_flat.tolist()))
            
            acc = accuracy_score(label_flat, preds_flat)
            
            unique_labels = set(label_flat.tolist())
            unique_preds = set(preds_flat.tolist())
            
            try:
                if unique_labels == {0} and unique_preds == {0}:
                    prec = rec = f1 = 1.0
                elif unique_labels == {1} and unique_preds == {1}:
                    prec = rec = f1 = 1.0
                else:
                    prec = precision_score(label_flat, preds_flat, pos_label=1, zero_division=0)
                    rec = recall_score(label_flat, preds_flat, pos_label=1, zero_division=0)
                    f1 = f1_score(label_flat, preds_flat, pos_label=1, zero_division=0)
            except Exception as e:
                print(f"⚠️ Error on sequence {name}: {e}")
                prec = rec = f1 = 0.0
            
            results.append({
                "name": name,
                "length": len(label_flat),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_domains": pred_domains,
                "true_domains": true_domains,
                "pred_labels": pred_str,
                "true_labels": label_str,
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Per-sequence test results saved to: {output_csv}")
    return df

# ----------------------------
# 6. main process
# ----------------------------
def main():
    cfg = args.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load embeddings
    part_names = [part.strip() for part in args.part_names.split(",")]
    embeddings_dict = {}
    labels_dict = {}

    all_train_names, all_val_names, all_test_names = [], [], []
    for part in part_names:
        npz_file = os.path.join(cfg.embedding_dir, f"{part}_embeddings.npz")
        label_dir = os.path.join(cfg.label_dir, part)
        data = np.load(npz_file)
        names_in_file = list()
        for name in data.files:         
            label_file = os.path.join(label_dir, f"{name}_domain.txt")
            label = load_label(label_file)
            if label.shape[0] != data[name].shape[0]:
                print(f"⚠️ length mismatch: {part}-{name} - label length {label.shape[0]} while emb length {data[name].shape[0]}")
                continue
            embeddings_dict[name] = data[name]  # [L, D]
            labels_dict[name] = label
            names_in_file.append(name)

        train, temp = train_test_split(names_in_file, test_size=0.2, random_state=cfg.random_seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=cfg.random_seed)
        
        all_train_names.extend(train)
        all_val_names.extend(val)
        all_test_names.extend(test)

        print(f"😄 File {os.path.basename(npz_file)}: train={len(train)}, val={len(val)}, test={len(test)}")
   
    print(f"✅ Loaded {len(embeddings_dict)} sequences from embeddings. {len(labels_dict)} sequences from labels.")

    
    # build dataset
    train_dataset = ProteinTokenDataset(all_train_names, embeddings_dict, labels_dict)
    val_dataset = ProteinTokenDataset(all_val_names, embeddings_dict, labels_dict)
    test_dataset = ProteinTokenDataset(all_test_names, embeddings_dict, labels_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print("✅ Dataset ok!")
    
    # model initialization
    model = TokenClassifier(
        hidden_size=cfg.hidden_size,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learing_rate)
    
    # training
    best_f1 = 0
    use_lstm = "LSTM" 
    best_model_path = os.path.join(cfg.output_dir, f"best_model_{use_lstm}_e{cfg.epochs}.pth")
    final_model_path = os.path.join(cfg.output_dir, f"final_model_{use_lstm}_e{cfg.epochs}.pth") 
    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
    
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved at epoch {cfg.epochs}")

    # final and f1-best model test
    test_metrics1 = evaluate(model, test_loader, device)
    print("🧪 Final Test Results:", test_metrics1)
    test_details_file = os.path.join(cfg.output_dir, f"test_results_{use_lstm}_e{cfg.epochs}.csv")
    evaluate_per_sequence_and_save(model, test_loader, device, test_details_file)
    judge_intervals(test_details_file)

    model.load_state_dict(torch.load(best_model_path))
    test_metrics2 = evaluate(model, test_loader, device)
    print("🧪 Best Test Results:", test_metrics2)
    
    # save results
    with open(os.path.join(cfg.output_dir, f"test_results_{use_lstm}_e{cfg.epochs}.txt"), "w") as f:
        f.write("final results:\n")
        f.write(str(test_metrics1))
        f.write("\nbest results:\n")
        f.write(str(test_metrics2))

    test_details_file = os.path.join(cfg.output_dir, f"test_results_{use_lstm}_e{cfg.epochs}_best.csv")
    evaluate_per_sequence_and_save(model, test_loader, device, test_details_file)
    judge_intervals(test_details_file)
    
    print(f"✅ All done! Results saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()