import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import os

# Cấu hình giao diện đẹp chuẩn báo Q1
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

def calculate_extended_metrics(y_true, y_pred, method_name):
    """
    Tính toán các chỉ số chi tiết: Acc, Precision, Recall, F1, FAR
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # Macro average để cân bằng giữa các lớp
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Tính False Alarm Rate (FAR) trung bình
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        far_per_class = FP / (FP + TN)
        far_per_class = np.nan_to_num(far_per_class)
    
    avg_far = np.mean(far_per_class)
    
    metrics_dict = {
        'Method': method_name,
        'Accuracy': round(accuracy * 100, 2),
        'Detection Rate (Recall)': round(recall * 100, 2),
        'Precision': round(precision * 100, 2),
        'F1-Score': round(f1 * 100, 2),
        'False Alarm Rate (FAR)': round(avg_far * 100, 2)
    }
    return metrics_dict, cm

def save_results(results_dict, filename="experiment_results.csv"):
    df = pd.DataFrame(results_dict)
    df.to_csv(f"results/{filename}", index=False)
    print(f"[INFO] Results saved to results/{filename}")

# --- CÁC HÀM VẼ HÌNH ---

def plot_confusion_matrix(cm, classes, method_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"results/Confusion_Matrix_{method_name}.png", dpi=300)
    plt.close()

def plot_performance_metrics(df):
    if df is None or df.empty: return
    metrics = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score']
    # Chỉ lấy các cột có trong dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    df_melted = df.melt(id_vars=['Method'], value_vars=available_metrics, 
                        var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Method', data=df_melted, palette="viridis")
    plt.title("Comparative Performance Analysis", fontsize=16, fontweight='bold')
    plt.ylim(0, 110) # Để scale đẹp
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, title='')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), 
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig("results/Fig2_Performance_Metrics.png", dpi=300)
    plt.close()

def plot_trust_evolution(df):
    if df is None or df.empty: return
    plt.figure(figsize=(12, 7))
    rounds = df['Round']
    threshold = df['Threshold']
    
    # Vẽ đường Threshold
    plt.plot(rounds, threshold, label='Adaptive Threshold', color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    
    client_cols = [c for c in df.columns if 'Client_' in c]
    labeled_good, labeled_bad, labeled_suspicious = False, False, False
    
    for client in client_cols:
        score_trend = df[client]
        final_score = score_trend.iloc[-1]
        
        if final_score < 0.2: 
            label = 'Malicious Clients (Blocked)' if not labeled_bad else ""
            plt.plot(rounds, score_trend, color='red', alpha=0.8, linewidth=2, label=label)
            labeled_bad = True
        elif final_score > 0.8:
            label = 'Trusted Clients' if not labeled_good else ""
            plt.plot(rounds, score_trend, color='green', alpha=0.3, linewidth=1, label=label)
            labeled_good = True
        else:
            label = 'Low-Quality Clients' if not labeled_suspicious else ""
            plt.plot(rounds, score_trend, color='orange', linestyle=':', linewidth=2, label=label)
            labeled_suspicious = True

    plt.title("Zero Trust Dynamics: Isolation of Malicious Nodes", fontsize=16, fontweight='bold')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Trust Score")
    plt.axhline(y=0.4, color='gray', linestyle='-', alpha=0.3)
    plt.legend(loc='center right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/Fig4_Trust_Evolution.png", dpi=300)
    plt.close()

def plot_efficiency(df):
    if df is None or df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'False Alarm Rate (FAR)' in df.columns:
        sns.barplot(x='Method', y='False Alarm Rate (FAR)', data=df, ax=axes[0], palette="Reds_d")
        axes[0].set_title("False Alarm Rate (Lower is Better)", fontweight='bold')
        axes[0].set_ylabel("FAR (%)")
        for p in axes[0].patches:
            if p.get_height() > 0:
                axes[0].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom')

    if 'Training Time (s)' in df.columns:
        sns.barplot(x='Method', y='Training Time (s)', data=df, ax=axes[1], palette="Blues_d")
        axes[1].set_title("Computational Cost (Training Time)", fontweight='bold')
        axes[1].set_ylabel("Time (seconds)")
        for p in axes[1].patches:
            if p.get_height() > 0:
                axes[1].annotate(f'{p.get_height():.1f}s', (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("results/Fig3_Efficiency_Analysis.png", dpi=300)
    plt.close()

def plot_robustness(df):
    if df is None or df.empty: return
    # Chuyển dữ liệu sang dạng long để vẽ lineplot
    df_melted = df.melt(id_vars=['Poison_Ratio'], var_name='Method', value_name='Accuracy')
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Poison_Ratio', y='Accuracy', hue='Method', data=df_melted, 
                 style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
    plt.title("Robustness against Poisoning Attacks (0% to 90%)", fontsize=14, fontweight='bold')
    plt.xlabel("Poisoning Ratio")
    plt.ylabel("Global Accuracy (%)")
    plt.tight_layout()
    plt.savefig("results/Fig5_Robustness_Total.png", dpi=300)
    plt.close()

def plot_convergence(results_dict, title="Convergence Analysis"):
    plt.figure(figsize=(10, 6))
    for method, acc_list in results_dict.items():
        plt.plot(acc_list, label=method, linewidth=2)
    plt.title(title)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Global Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/convergence_{title.replace(' ', '_')}.png", dpi=300)
    plt.close()
# --- BỔ SUNG: Hàm xuất Confusion Matrix ra CSV ---
def save_confusion_matrix_csv(cm, classes, method_name):
    """
    Lưu ma trận nhầm lẫn ra file CSV để dễ phân tích trong Excel.
    """
    import pandas as pd
    # Tạo DataFrame với nhãn dòng và cột là tên lớp (DoS, Normal...)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Lưu file
    filename = f"results/Confusion_Matrix_{method_name}.csv"
    df_cm.to_csv(filename)
    print(f"[INFO] Saved Confusion Matrix CSV: {filename}")