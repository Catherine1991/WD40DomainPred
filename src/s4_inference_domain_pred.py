# inference.py
import os
import argparse
import torch
import pandas as pd
from src.s3_train_WDP import TokenClassifier, ProteinTokenDataset, apply_rules_and_extract_domains, is_prediction_valid
from src.judge import judge_intervals, evaluate_and_show_results
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained WD40 domain prediction model")
    parser.add_argument("--model_path", type=str, default=".model/final_model_LSTM_e51.pth", help="Path to saved model .pth file")
    parser.add_argument("--embedding_dir", type=str, default="./output/embeddings",
                        help="Directory containing .npz embedding files")
    parser.add_argument("--label_file", type=str, default="./data/inference_labels.csv", help="inference labels")
    parser.add_argument("--train_part", type=str, default="positive,negative,negative2",
                        help="Which part to infer on")
    parser.add_argument("--inference_part", type=str, default="inf_part1,inf_part2",
                        help="Which part to infer on")
    parser.add_argument("--output_csv", type=str, default="./output/inference_output/inference_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def parse_uniprot_domains(csv_path):
    """
    parse CSV file
    - key: UniProt_ID 
    - value: 
        - if "start-end" is "0-0", return []
        - else "start-end" seperated by '-' return [start-1, end-1]
    
    Args:
        csv_path (str): CSV file path
    
    Returns:
        dict: {UniProt_ID: [start-1, end-1] or []}
    """
    df = pd.read_csv(csv_path)
    
    if 'UniProt_ID' not in df.columns or 'start-end' not in df.columns:
        raise ValueError("CSV must contain columns 'UniProt_ID' and 'start-end'")
    
    result = {}
    for _, row in df.iterrows():
        uniprot_id = str(row['UniProt_ID'])
        se_str = str(row['start-end']).strip()
        
        if se_str == "0-0":
            result[uniprot_id] = []
        else:
            try:
                parts = se_str.split('-')
                if len(parts) != 2:
                    # raise ValueError(f"Invalid format in 'start-end': {se_str}")
                    # multi domains skipped
                    continue
                start, end = int(parts[0]), int(parts[-1])
                result[uniprot_id] = [start - 1, end - 1]
            except Exception as e:
                raise ValueError(f"Error parsing 'start-end' value '{se_str}' for UniProt_ID {uniprot_id}: {e}")
    
    return result


def load_embeddings_and_labels(embedding_dir, label_file, parts, train_set):
    embeddings_dict = {}
    labels_dict = {}
    seq_names = []

    label_exists = parse_uniprot_domains(label_file)
    for part in parts:        
        npz_file = os.path.join(embedding_dir, f"{part}_embeddings.npz")
        if not os.path.exists(npz_file):
            print(f"⚠️ Skipping {part}: {npz_file} not found")
            continue
        data = np.load(npz_file)
            
        for name in data.files:
            try:
                if name not in label_exists or name in train_set:
                    # if sample doesn't have label or has already trained, skip...
                    continue
                emb = data[name]
                seq_len = emb.shape[0]
                labels = np.zeros(seq_len, dtype=int)
                if label_exists[name]:
                    start, end = label_exists[name]
                    labels[start:end+1] = 1

                embeddings_dict[name] = emb
                labels_dict[name] = labels
                seq_names.append(name)
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
                continue

    return seq_names, embeddings_dict, labels_dict

def evaluate_per_sequence_and_save(model, dataloader, device, output_csv, use_filter=True, valid_thr:int=5):
    model.eval()
    results = []
    
    with torch.no_grad():
        for name, emb, label in tqdm(dataloader, desc="Testing per sequence"):
            emb = emb.to(device)
            label_np = label.numpy()
            
            logits = model(emb)  # [L]
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            # Ensure 1D
            label_flat = label_np.flatten()
            preds_flat = preds.flatten()
        
            if use_filter:
                res_rules = apply_rules_and_extract_domains(preds_flat, label_flat)
                preds_flat = res_rules["preds_rule"]
                pred_domains = res_rules["pred_domains"]  # e.g., [100, 200] or []
                true_domains = res_rules["true_domains"]  # e.g., [98, 202] or []
 
                if true_domains:
                    true_label = 1
                    if is_prediction_valid(pred_domains, true_domains, valid_thr):
                        pred_label = 1
                    else:
                        pred_label = 0
                else:
                    true_label = 0
                    pred_label = 1 if pred_domains else 0

            # Convert to string for CSV
            label_str = ''.join(map(str, label_flat.tolist()))
            pred_str = ''.join(map(str, preds_flat.tolist()))
            
            # Compute token-level metrics
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
            
            # Build result dict
            result = {
                "name": name,
                "length": len(label_flat),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            }

            if use_filter:
                result.update({
                    "true_label": true_label,      # 1 if true domain exists, else 0
                    "pred_label": pred_label,      # 1 if correctly predicted domain (or falsely predicted when none exists)
                    "pred_domains": pred_domains,  # keep as list, e.g., [100, 200] or []
                    "true_domains": true_domains,  # keep as list
                })
            result.update({"pred_labels": pred_str,
                            "true_labels": label_str,})
            
            results.append(result)
    
    # Save to CSV — pandas handles list columns by converting to string like "[100, 200]"
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Per-sequence test results saved to: {output_csv}")
    return df


def main():
    args = parse_args()

    # parts & get uniprot_id already trained
    parts_1st = args.train_part.split(",")
    parts_2nd = args.inference_part.split(",")

    already_trained = set()
    for part in parts_1st:
        npz_file = os.path.join(args.embedding_dir, f"{part}_embeddings.npz")
        data = np.load(npz_file)
        for name in data.files:
            already_trained.add(name.strip())
    print(f"👌 {len(already_trained)} proteins already trained.")

    # Load data
    print("🔍 Loading embeddings and labels...")
    seq_names, embeddings_dict, labels_dict = load_embeddings_and_labels(
        args.embedding_dir, args.label_file, parts_2nd, already_trained
    )
    
    if not seq_names:
        raise ValueError("No valid sequences loaded!")
    else:
        print(f"✅️ totally got {len(seq_names)} sequences!")

    dataset = ProteinTokenDataset(seq_names, embeddings_dict, labels_dict)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = TokenClassifier(
        hidden_size=1280,
        use_bilstm=True,
        lstm_hidden=256,
        dropout=0.3
    ).to(args.device)

    # Load weights
    print(f"📥 Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    results = []
    evaluate_per_sequence_and_save(model, dataloader, args.device, args.output_csv, use_filter=True)

    judge_intervals(args.output_csv)
    evaluate_and_show_results(args.output_csv)

if __name__ == "__main__":
    main()
    