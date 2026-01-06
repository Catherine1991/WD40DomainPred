import pandas as pd
import sys
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def is_prediction_valid(pred_domains, true_domains, valid_thr=5):
    return pred_domains and abs(pred_domains[0]-true_domains[0]) <= valid_thr and abs(pred_domains[-1]-true_domains[-1]) <= valid_thr


def judge_flags(csv_path: str,):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"error '{csv_path}' not exists", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print("error file is empty", file=sys.stderr)
        return
    except Exception as e:
        print(f"error while reading file: {e}", file=sys.stderr)
        return

    metrics_per_class = {}

    supports = []

    for cls in [0, 1]:
        tp = ((df['true_label'] == cls) & (df['pred_label'] == cls)).sum()
        fp = ((df['true_label'] != cls) & (df['pred_label'] == cls)).sum()
        fn = ((df['true_label'] == cls) & (df['pred_label'] != cls)).sum()
        support = (df['true_label'] == cls).sum()
        supports.append(support)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_per_class[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
            'correct': tp
        }

    print(">>> Per-class metrics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<8} {'Correct'}")
    for cls in [0, 1]:
        m = metrics_per_class[cls]
        print(f"{cls:<6} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<8} {m['correct']}")

    total_samples = len(df)

    accuracy = (df['true_label'] == df['pred_label']).sum() / total_samples

    macro_precision = (metrics_per_class[0]['precision'] + metrics_per_class[1]['precision']) / 2
    macro_recall = (metrics_per_class[0]['recall'] + metrics_per_class[1]['recall']) / 2
    macro_f1 = (metrics_per_class[0]['f1-score'] + metrics_per_class[1]['f1-score']) / 2

    weighted_f1 = (
        metrics_per_class[0]['f1-score'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['f1-score'] * metrics_per_class[1]['support']
    ) / total_samples

    weighted_recall = (
        metrics_per_class[0]['recall'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['recall'] * metrics_per_class[1]['support']
    ) / total_samples

    print("\n>>> Overall metrics:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro-Precision:   {macro_precision:.4f}")
    print(f"Macro-Recall:      {macro_recall:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")
    print(f"Weighted-Recall:   {weighted_recall:.4f}")


def judge_intervals(csv_path: str, valid_thr:int=5):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"error '{csv_path}' not exists", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print("error file is empty", file=sys.stderr)
        return
    except Exception as e:
        print(f"error while reading file: {e}", file=sys.stderr)
        return


    def parse_and_label(true_domain, pred_domain, valid_thr=5):
        try:
            if pd.isna(true_domain) or true_domain.strip() == '' or pd.isna(pred_domain) or pred_domain.strip():
                return None, None
            true_dm = ast.literal_eval(true_domain)
            pred_dm = ast.literal_eval(pred_domain)
            if true_dm and pred_dm:
                if is_prediction_valid(pred_dm, true_dm, valid_thr):
                    return 1,1
                else:
                    return 1,0
            elif true_dm:
                return 1, 0
            elif pred_dm:
                return 0, 1
            else:
                return 0, 0
        except (ValueError, SyntaxError):
            return None, None

    true_labels = []
    pred_labels = []

    for _, row in df.iterrows():
        true_dom = row['true_domains']
        pred_dom = row['pred_domains']

        try:
            if pd.isna(true_dom) or pd.isna(pred_dom):
                continue
            true_str = str(true_dom).strip()
            pred_str = str(pred_dom).strip()
            if true_str == '' or pred_str == '':
                continue

            true_dm = ast.literal_eval(true_str)
            pred_dm = ast.literal_eval(pred_str)

            if not isinstance(true_dm, list) or not isinstance(pred_dm, list):
                continue

        except (ValueError, SyntaxError):
            continue

        if true_dm and pred_dm:
            if is_prediction_valid(pred_dm, true_dm, valid_thr):
                y_true, y_pred = 1, 1
            else:
                y_true, y_pred = 1, 0
        elif true_dm:  
            y_true, y_pred = 1, 0
        elif pred_dm:   
            y_true, y_pred = 0, 1
        else:           
            y_true, y_pred = 0, 0

        true_labels.append(y_true)
        pred_labels.append(y_pred)

    if len(true_labels) == 0:
        print("no valid samples", file=sys.stderr)
        return


    metrics_per_class = {}
    total_samples = len(true_labels)

    for cls in [0, 1]:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support = sum(1 for t in true_labels if t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_per_class[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
            'correct': tp
        }

    # per-class
    print(">>> Per-class metrics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<8} {'Correct'}")
    for cls in [0, 1]:
        m = metrics_per_class[cls]
        print(f"{cls:<6} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<8} {m['correct']}")

    # whole
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / total_samples

    # Macro 
    macro_precision = (metrics_per_class[0]['precision'] + metrics_per_class[1]['precision']) / 2
    macro_recall = (metrics_per_class[0]['recall'] + metrics_per_class[1]['recall']) / 2
    macro_f1 = (metrics_per_class[0]['f1-score'] + metrics_per_class[1]['f1-score']) / 2

    # Weighted
    weighted_f1 = (
        metrics_per_class[0]['f1-score'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['f1-score'] * metrics_per_class[1]['support']
    ) / total_samples

    weighted_precision = (
        metrics_per_class[0]['precision'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['precision'] * metrics_per_class[1]['support']
    ) / total_samples

    weighted_recall = (
        metrics_per_class[0]['recall'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['recall'] * metrics_per_class[1]['support']
    ) / total_samples

    print("\n>>> Overall metrics:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro-Precision:   {macro_precision:.4f}")
    print(f"Macro-Recall:      {macro_recall:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-Precision:{weighted_precision:.4f}")
    print(f"Weighted-Recall:   {weighted_recall:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

def evaluate_from_csv(csv_path):

    def safe_literal_eval(s):
        try:
            return ast.literal_eval(s) if pd.notna(s) and str(s).strip() else []
        except:
            return []

    def apply_domain_mask(label_str, domain_list):
        L = len(label_str)
        mask = [0] * L
        if domain_list and len(domain_list) == 2:
            start, end = domain_list
            start = max(0, min(start, L - 1))
            end = max(start, min(end, L - 1))
            for i in range(start, end + 1):
                mask[i] = 1
        return mask

    def compute_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        
        # Class-wise (for class 0 and 1)
        prec = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
        
        # Macro-average as "overall" metrics
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return {
            "overall_accuracy": round(acc, 4),
            "overall_precision": round(macro_prec, 4),   
            "overall_recall": round(macro_rec, 4),       
            "overall_f1": round(macro_f1, 4),            
            "class_0_precision": round(prec[0], 4),
            "class_0_recall": round(rec[0], 4),
            "class_0_f1": round(f1[0], 4),
            "class_1_precision": round(prec[1], 4),
            "class_1_recall": round(rec[1], 4),
            "class_1_f1": round(f1[1], 4),
        }

    df = pd.read_csv(csv_path)

    all_true = []
    all_pred_orig = []
    all_pred_filt = []

    for _, row in df.iterrows():
        true_str = str(row['true_labels']).strip()
        pred_str = str(row['pred_labels']).strip()

        if len(true_str) != len(pred_str):
            raise ValueError(f"Length mismatch in row (name={row.get('name', 'unknown')}): "
                             f"true={len(true_str)}, pred={len(pred_str)}")

        y_true = [int(c) for c in true_str]
        y_pred_orig = [int(c) for c in pred_str]

        all_true.extend(y_true)
        all_pred_orig.extend(y_pred_orig)

        # mask
        domains = safe_literal_eval(row['pred_domains'])
        y_pred_filt = apply_domain_mask(pred_str, domains)
        all_pred_filt.extend(y_pred_filt)

    metrics_original = compute_metrics(all_true, all_pred_orig)
    metrics_filtered = compute_metrics(all_true, all_pred_filt)

    return {
        "original": metrics_original,
        "filtered": metrics_filtered
    }

def evaluate_and_show_results(csv_path):
    results = evaluate_from_csv(csv_path)

    def print_metrics(title, metrics):
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Overall Precision (macro): {metrics['overall_precision']:.4f}")
        print(f"Overall Recall    (macro): {metrics['overall_recall']:.4f}")
        print(f"Overall F1-score  (macro): {metrics['overall_f1']:.4f}")
        print()
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)
        print(f"{'0 (negative)':<8} {metrics['class_0_precision']:<10} {metrics['class_0_recall']:<10} {metrics['class_0_f1']:<10}")
        print(f"{'1 (positive)':<8} {metrics['class_1_precision']:<10} {metrics['class_1_recall']:<10} {metrics['class_1_f1']:<10}")

    print_metrics("Original Prediction", results["original"])
    print_metrics("After Applying pred_domains Mask", results["filtered"])

    return results


if __name__ == "__main__":

    csv_file = "/data/ShowMakerTAT/tools/SaProt/output/protein_domain_pred/test_results_LSTM_e51.csv"
    # csv_file = "/data/ShowMakerTAT/tools/SaProt/output/protein_domain_pred/test_results_cnn_bilstm.csv"
    csv_file = "/data/ShowMakerTAT/tools/SaProt/output/inference_2nd/inference_results_1231_new.csv"
    csv_file = "/data/ShowMakerTAT/tools/SaProt/output/inference_2nd/inference_results_0105.csv"
    # judge_flags(csv_file)
    # judge_intervals(csv_file, 3)
    evaluate_and_show_results(csv_file)


    csv_file = "/data/ShowMakerTAT/tools/SaProt/output/protein_domain_pred/test_results_LSTM_e51_best.csv"
    # csv_file = "/data/ShowMakerTAT/tools/SaProt/output/protein_domain_pred/test_results_cnn_bilstm.csv"
    # judge_flags(csv_file)
