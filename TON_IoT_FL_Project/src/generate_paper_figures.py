import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình giao diện đẹp chuẩn báo Q1
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif' # Font có chân trông học thuật hơn

def load_data():
    if not os.path.exists("results"):
        print("[ERROR] Folder 'results' not found!")
        return None, None, None
        
    try:
        perf_df = pd.read_csv("results/Final_Performance_Table.csv")
        trust_df = pd.read_csv("results/trust_scores_poison_0.2.csv")
        robust_df = pd.read_csv("results/robustness_analysis.csv")
        return perf_df, trust_df, robust_df
    except Exception as e:
        print(f"[ERROR] Could not load CSV files: {e}")
        return None, None, None

def plot_performance_metrics(df):
    """Vẽ Accuracy, Recall, Precision, F1 trên cùng 1 biểu đồ"""
    # Chuyển đổi dữ liệu sang dạng 'long' để vẽ seaborn
    metrics = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score']
    df_melted = df.melt(id_vars=['Method'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Method', data=df_melted, palette="viridis")
    
    plt.title("Performance Comparison across Metrics", fontsize=14, fontweight='bold')
    plt.ylabel("Percentage (%)")
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.legend(loc='lower right', title='Framework')
    
    # Thêm số lên đầu cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig("results/Fig2_Performance_Metrics.png", dpi=300)
    print("[INFO] Saved Fig2_Performance_Metrics.png")

def plot_efficiency(df):
    """Vẽ FAR và Training Time (2 biểu đồ con)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. False Alarm Rate (Càng thấp càng tốt)
    sns.barplot(x='Method', y='False Alarm Rate (FAR)', data=df, ax=axes[0], palette="Reds_d")
    axes[0].set_title("False Alarm Rate (Lower is Better)", fontweight='bold')
    axes[0].set_ylabel("FAR (%)")
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom')

    # 2. Training Time (Càng thấp càng tốt)
    sns.barplot(x='Method', y='Training Time (s)', data=df, ax=axes[1], palette="Blues_d")
    axes[1].set_title("Computational Cost (Training Time)", fontweight='bold')
    axes[1].set_ylabel("Time (seconds)")
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f}s', (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("results/Fig3_Efficiency_Analysis.png", dpi=300)
    print("[INFO] Saved Fig3_Efficiency_Analysis.png")

def plot_trust_evolution(df):
    """Vẽ diễn biến Trust Score (Minh chứng Zero Trust)"""
    plt.figure(figsize=(10, 6))
    
    # Xác định Client tốt và Client xấu (Dựa trên dữ liệu bạn gửi)
    # Client 1 là Hacker (bị drop thê thảm), Client 0 là người tốt
    
    rounds = df['Round']
    threshold = df['Threshold']
    
    # Vẽ đường Threshold
    plt.plot(rounds, threshold, label='Dynamic Threshold', color='black', linestyle='--', linewidth=2)
    
    # Vẽ Client Tốt (Client 0)
    plt.plot(rounds, df['Client_0'], label='Benign Client (Normal)', color='green', marker='o', markevery=2)
    
    # Vẽ Client Xấu (Client 1) - Kẻ tấn công
    plt.plot(rounds, df['Client_1'], label='Malicious Client (Attacker)', color='red', marker='x', linewidth=2.5)
    
    # Vẽ Client Nghi ngờ (Client 2 - Acc thấp nhưng chưa hẳn là hacker)
    if 'Client_2' in df.columns:
         plt.plot(rounds, df['Client_2'], label='Low-Quality Client', color='orange', linestyle=':')

    plt.title("Trust Score Evolution & Zero Trust Isolation", fontsize=14, fontweight='bold')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Trust Score (0.0 - 1.0)")
    plt.axhline(y=0.4, color='gray', linestyle='-', alpha=0.3) # Sàn an toàn
    
    # Chú thích vùng bị chặn
    plt.text(3.5, 0.25, "Zero Trust Blocked!", color='red', fontweight='bold', fontsize=12)
    plt.arrow(3.2, 0.22, 0, -0.15, head_width=0.2, head_length=0.05, fc='red', ec='red')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/Fig4_Trust_Evolution.png", dpi=300)
    print("[INFO] Saved Fig4_Trust_Evolution.png")

def plot_robustness(df):
    """Vẽ Robustness (Acc vs Poison Ratio)"""
    # Chuyển bảng robustness thành dạng long để vẽ
    # df đang là: Poison_Ratio, FedAvg, DynamicFuzzy
    df_melted = df.melt(id_vars=['Poison_Ratio'], var_name='Method', value_name='Accuracy')
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='Poison_Ratio', y='Accuracy', hue='Method', data=df_melted, 
                 style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
    
    plt.title("Robustness against Poisoning Attacks", fontsize=14, fontweight='bold')
    plt.xlabel("Poisoning Ratio (Percentage of Malicious Clients)")
    plt.ylabel("Global Accuracy (%)")
    plt.xticks([0.0, 0.2]) # Chỉ hiện các mốc có dữ liệu
    
    plt.tight_layout()
    plt.savefig("results/Fig5_Robustness.png", dpi=300)
    print("[INFO] Saved Fig5_Robustness.png")

def main():
    perf_df, trust_df, robust_df = load_data()
    if perf_df is not None:
        plot_performance_metrics(perf_df)
        plot_efficiency(perf_df)
    
    if trust_df is not None:
        plot_trust_evolution(trust_df)
        
    if robust_df is not None:
        plot_robustness(robust_df)
        
    print("\n[SUCCESS] All figures generated in 'results/' folder.")

if __name__ == "__main__":
    main()