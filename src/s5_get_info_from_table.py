import pandas as pd
import sys
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def is_prediction_valid(pred_domains, true_domains, valid_thr=5):
    return pred_domains and abs(pred_domains[0]-true_domains[0]) <= valid_thr and abs(pred_domains[-1]-true_domains[-1]) <= valid_thr


def judge_flags(csv_path: str,):
    """
    读取 CSV 文件，并遍历指定列的每一行。

    参数:
        csv_path (str): CSV 文件路径。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：文件 '{csv_path}' 不存在。", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print("错误：CSV 文件为空。", file=sys.stderr)
        return
    except Exception as e:
        print(f"读取 CSV 时发生错误: {e}", file=sys.stderr)
        return

    metrics_per_class = {}

    # 存储每类的 support 用于加权平均
    supports = []

    for cls in [0, 1]:
        # TP: 真实是 cls，预测也是 cls
        tp = ((df['true_label'] == cls) & (df['pred_label'] == cls)).sum()
        
        # FP: 真实不是 cls，但预测是 cls
        fp = ((df['true_label'] != cls) & (df['pred_label'] == cls)).sum()
        
        # FN: 真实是 cls，但预测不是 cls
        fn = ((df['true_label'] == cls) & (df['pred_label'] != cls)).sum()
        
        # Support: 真实为 cls 的样本数
        support = (df['true_label'] == cls).sum()
        supports.append(support)
        
        # 计算指标
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

    # ==============================
    # 打印：每个类别的详细结果
    # ==============================
    print(">>> Per-class metrics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<8} {'Correct'}")
    for cls in [0, 1]:
        m = metrics_per_class[cls]
        print(f"{cls:<6} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<8} {m['correct']}")

    # ==============================
    # 计算总体（汇总）指标
    # ==============================
    total_samples = len(df)

    # 总准确率
    accuracy = (df['true_label'] == df['pred_label']).sum() / total_samples

    # Macro 平均（各类指标的算术平均）
    macro_precision = (metrics_per_class[0]['precision'] + metrics_per_class[1]['precision']) / 2
    macro_recall = (metrics_per_class[0]['recall'] + metrics_per_class[1]['recall']) / 2
    macro_f1 = (metrics_per_class[0]['f1-score'] + metrics_per_class[1]['f1-score']) / 2

    # Weighted 平均（按 support 加权）
    weighted_f1 = (
        metrics_per_class[0]['f1-score'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['f1-score'] * metrics_per_class[1]['support']
    ) / total_samples

    weighted_recall = (
        metrics_per_class[0]['recall'] * metrics_per_class[0]['support'] +
        metrics_per_class[1]['recall'] * metrics_per_class[1]['support']
    ) / total_samples

    # ==============================
    # 打印：总体指标
    # ==============================
    print("\n>>> Overall metrics:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro-Precision:   {macro_precision:.4f}")
    print(f"Macro-Recall:      {macro_recall:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")
    print(f"Weighted-Recall:   {weighted_recall:.4f}")
    
    # pos_sum, neg_sum = 0, 0
    # pos_val, neg_val = 0, 0
    # for f1, f2 in zip(df['true_label'], df['pred_label']):
    #     if f1 == 1:
    #         pos_sum += 1
    #         if f2 == 1:
    #             pos_val += 1
    #     else:
    #         neg_sum += 1
    #         if f2 == 1:
    #             neg_val += 1
    # print("============ TEST RESULTS with Error Tolerance 5 ============")
    # print(f"\tpositive {pos_val}/{pos_sum}={pos_val/pos_sum:.4f}")
    # print(f"\tnegative {neg_val}/{neg_sum}={neg_val/neg_sum:.4f}")
    # print(f"\ttotally  {pos_val+neg_val}/{pos_sum+neg_sum}={(pos_val+neg_val)/(neg_sum+pos_sum):.4f}")
    # print("=============================================================")


def judge_intervals(csv_path: str, valid_thr:int=5):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：文件 '{csv_path}' 不存在。", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print("错误：CSV 文件为空。", file=sys.stderr)
        return
    except Exception as e:
        print(f"读取 CSV 时发生错误: {e}", file=sys.stderr)
        return


    def parse_and_label(true_domain, pred_domain, valid_thr=5):
        try:
            # 处理 NaN 或空值
            if pd.isna(true_domain) or true_domain.strip() == '' or pd.isna(pred_domain) or pred_domain.strip():
                return None, None
            # 使用 ast.literal_eval 安全解析字符串为 list
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
            # 如果解析失败，视为无结构域
            return None, None

    # 存储有效样本的标签
    true_labels = []
    pred_labels = []

    # 遍历每一行
    for _, row in df.iterrows():
        true_dom = row['true_domains']
        pred_dom = row['pred_domains']

        try:
            # 处理 NaN 或空字符串
            if pd.isna(true_dom) or pd.isna(pred_dom):
                continue
            true_str = str(true_dom).strip()
            pred_str = str(pred_dom).strip()
            if true_str == '' or pred_str == '':
                continue

            true_dm = ast.literal_eval(true_str)
            pred_dm = ast.literal_eval(pred_str)

            # 确保解析结果是 list
            if not isinstance(true_dm, list) or not isinstance(pred_dm, list):
                continue

        except (ValueError, SyntaxError):
            continue  # 跳过无法解析的行

        # 核心：调用你的逻辑生成标签
        if true_dm and pred_dm:
            if is_prediction_valid(pred_dm, true_dm, valid_thr):
                y_true, y_pred = 1, 1
            else:
                y_true, y_pred = 1, 0
        elif true_dm:          # 有真结构域，但没预测出
            y_true, y_pred = 1, 0
        elif pred_dm:          # 无真结构域，但预测出了（假阳性）
            y_true, y_pred = 0, 1
        else:                  # 都没有
            y_true, y_pred = 0, 0

        true_labels.append(y_true)
        pred_labels.append(y_pred)

    # 转为 DataFrame 便于后续计算（或直接用列表）
    if len(true_labels) == 0:
        print("警告：没有有效样本可用于评估。", file=sys.stderr)
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

    # 打印 per-class
    print(">>> Per-class metrics:")
    print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<8} {'Correct'}")
    for cls in [0, 1]:
        m = metrics_per_class[cls]
        print(f"{cls:<6} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<8} {m['correct']}")

    # 总体指标
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / total_samples

    # Macro 平均（各类指标的算术平均）
    macro_precision = (metrics_per_class[0]['precision'] + metrics_per_class[1]['precision']) / 2
    macro_recall = (metrics_per_class[0]['recall'] + metrics_per_class[1]['recall']) / 2
    macro_f1 = (metrics_per_class[0]['f1-score'] + metrics_per_class[1]['f1-score']) / 2

    # Weighted 平均（按 support 加权）
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

    # 打印：总体指标（完整版）
    print("\n>>> Overall metrics:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro-Precision:   {macro_precision:.4f}")
    print(f"Macro-Recall:      {macro_recall:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-Precision:{weighted_precision:.4f}")
    print(f"Weighted-Recall:   {weighted_recall:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

def evaluate_from_csv(csv_path):
    """
    从 CSV 文件中读取 pred_labels、true_labels 和 pred_domains，
    计算两组 token-level 评估指标：
      1. 原始 pred_labels vs true_labels
      2. 应用 pred_domains 区间掩码后的 pred_labels vs true_labels

    要求 CSV 列：
      - 'pred_labels': 字符串，如 "001100"
      - 'true_labels': 字符串，如 "001110"
      - 'pred_domains': 字符串形式的列表，如 "[100, 200]" 或 "[]"

    返回:
        dict: {
            "original": {overall_accuracy, class_0/1 metrics...},
            "filtered": {overall_accuracy, class_0/1 metrics...}
        }
    """
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
    # 读取数据
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

        # 应用 domain 掩码
        domains = safe_literal_eval(row['pred_domains'])
        y_pred_filt = apply_domain_mask(pred_str, domains)
        all_pred_filt.extend(y_pred_filt)

    # 计算指标
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
