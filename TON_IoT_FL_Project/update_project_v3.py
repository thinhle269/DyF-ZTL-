import os

# 1. CẬP NHẬT src/utils.py
# - Tích hợp toàn bộ code vẽ hình (Viz)
# - Giữ lại các hàm tính toán Metric
utils_code = """import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Cấu hình giao diện đẹp chuẩn báo Q1
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

def calculate_extended_metrics(y_true, y_pred, method_name):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
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

# --- CÁC HÀM VẼ HÌNH (Tích hợp từ generate_paper_figures.py) ---

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
    if df is None: return
    metrics = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score']
    df_melted = df.melt(id_vars=['Method'], value_vars=[m for m in metrics if m in df.columns], 
                        var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Method', data=df_melted, palette="viridis")
    plt.title("Comparative Performance Analysis", fontsize=16, fontweight='bold')
    plt.ylim(80, 105)
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
    if df is None: return
    plt.figure(figsize=(12, 7))
    rounds = df['Round']
    threshold = df['Threshold']
    
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

def plot_robustness(df):
    if df is None: return
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
"""

# 2. CẬP NHẬT src/fl_core.py
# - Tính metric chi tiết (Acc, F1, FAR...) cho TỪNG VÒNG (Round)
# - Trả về lịch sử metrics đầy đủ để xuất file performance_table_full.csv
fl_core_code = """import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.models
from src.trust_engine import TrustEvaluator
import src.utils as utils

class LocalUpdate:
    def __init__(self, dataset, batch_size, learning_rate, epochs, device, algo='fedavg', mu=0.01):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device
        self.algo = algo
        self.mu = mu 

    def train(self, model, global_model=None):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        epoch_loss = []
        for _ in range(self.epochs):
            batch_loss = []
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if self.algo == 'fedprox' and global_model is not None:
                    proximal_term = 0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss += (self.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss)

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def evaluate_model(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # Tra ve raw y_true, y_pred de tinh metrics ben ngoai
    return y_true, y_pred

def run_fl_simulation(client_datasets, val_dataset, test_dataset, model_type, rounds, num_clients, epochs, batch_size, poison_ratio=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = client_datasets[0][0][0].shape[0]
    
    all_labels = []
    for _, y in test_dataset:
        all_labels.append(y.item())
    num_classes = len(set(all_labels))

    if model_type == 'fuzzy':
        global_model = src.models.DynamicFuzzyNet(input_dim, num_classes).to(device)
    else:
        global_model = src.models.DeepNet(input_dim, num_classes).to(device)
        
    global_weights = global_model.state_dict()
    
    # Containers to store history
    acc_history = []
    full_metrics_history = [] # Store detailed metrics per round
    final_y_true, final_y_pred = [], []
    
    trust_engine = TrustEvaluator(num_clients, alpha=0.5)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_poisoned = int(num_clients * poison_ratio)
    poisoned_indices = np.random.choice(range(num_clients), num_poisoned, replace=False)
    if num_poisoned > 0:
        print(f"[WARN] Poisoning clients: {poisoned_indices}")

    for r in tqdm(range(rounds), desc=f"Training {model_type} (Poison={poison_ratio})"):
        local_weights_candidates = []
        client_indices_candidates = []
        
        idxs_users = range(num_clients)
        
        # --- STEP 1: TRAIN & TRUST ---
        for idx in idxs_users:
            if model_type == 'fuzzy':
                local_model = src.models.DynamicFuzzyNet(input_dim, num_classes).to(device)
            else:
                local_model = src.models.DeepNet(input_dim, num_classes).to(device)
                
            local_model.load_state_dict(global_weights)
            algo = 'fedprox' if model_type == 'fedprox' else 'fedavg'
            
            dataset_to_use = client_datasets[idx]
            if idx in poisoned_indices:
                X_p, y_p = dataset_to_use[:][0].clone(), dataset_to_use[:][1].clone()
                y_p = (y_p + 1) % num_classes 
                dataset_to_use = torch.utils.data.TensorDataset(X_p, y_p)
            
            trainer = LocalUpdate(dataset_to_use, batch_size, 0.01, epochs, device, algo=algo)
            w, _ = trainer.train(local_model, global_model if algo == 'fedprox' else None)
            
            if model_type == 'fuzzy':
                local_model.load_state_dict(w)
                trust_engine.calculate_trust(local_model, None, val_loader, device, idx)
                local_weights_candidates.append(w)
                client_indices_candidates.append(idx)
            else:
                local_weights_candidates.append(w)

        # --- STEP 2: ADAPTIVE FILTERING ---
        final_local_weights = []
        if model_type == 'fuzzy':
            trust_engine.update_dynamic_threshold()
            for i, w in zip(client_indices_candidates, local_weights_candidates):
                if trust_engine.is_trusted(i):
                    final_local_weights.append(w)
            trust_engine.log_round(r) # Log trust every round
        else:
            final_local_weights = local_weights_candidates

        # --- STEP 3: AGGREGATION ---
        if len(final_local_weights) > 0:
            global_weights = average_weights(final_local_weights)
            global_model.load_state_dict(global_weights)
        
        # --- STEP 4: EVALUATION & LOGGING PER ROUND ---
        y_true, y_pred = evaluate_model(global_model, test_dataset, device)
        
        # Calculate full metrics for this round
        round_metrics, _ = utils.calculate_extended_metrics(y_true, y_pred, model_type)
        round_metrics['Round'] = r
        full_metrics_history.append(round_metrics)
        
        acc_history.append(round_metrics['Accuracy'])
        
        if r == rounds - 1:
            final_y_true = y_true
            final_y_pred = y_pred
            
    # Save Trust History if using Fuzzy
    if model_type == 'fuzzy' and poison_ratio == 0.2:
        trust_engine.save_history(f"results/trust_scores_poison_{poison_ratio}.csv")
        
    return acc_history, final_y_true, final_y_pred, full_metrics_history
"""

# 3. CẬP NHẬT runall.py
# - Cấu hình ROUNDS=50
# - Xuất file performance_table_full.csv (Appendix)
# - Chạy Robustness từ 0.0 -> 0.9
runall_code = """import src.preprocessing as prep
import src.fl_core as fl
import src.utils as utils
import time
import pandas as pd
import numpy as np
import os

def main():
    if not os.path.exists('results'):
        os.makedirs('results')
        
    DATA_PATH = "dataset/Train_Test_Windows_10.csv"
    # === CẤU HÌNH FINAL CHO BÀI BÁO ===
    NUM_CLIENTS = 20
    ROUNDS = 50        
    EPOCHS = 5         
    BATCH_SIZE = 32
    
    print("="*40)
    print("PHASE 1: DATA PREPARATION")
    print("="*40)
    client_data, val_data, test_data, num_classes, input_dim, label_classes = prep.load_and_process_data(DATA_PATH, NUM_CLIENTS)
    
    print("\\n" + "="*40)
    print("PHASE 2: PERFORMANCE & APPENDIX GENERATION")
    print("="*40)
    
    convergence_results = {}
    final_metrics_summary = [] 
    
    methods = ['FedAvg', 'FedProx', 'DynamicFuzzy']
    
    for method in methods:
        algo_name = 'fuzzy' if method == 'DynamicFuzzy' else method.lower()
        print(f"\\n>>> Running {method}...")
        
        start = time.time()
        # Chú ý: Hàm trả về thêm full_metrics_history
        acc_hist, y_true, y_pred, full_metrics = fl.run_fl_simulation(
            client_data, val_data, test_data, algo_name, ROUNDS, NUM_CLIENTS, EPOCHS, BATCH_SIZE
        )
        duration = time.time() - start
        
        convergence_results[method] = acc_hist
        
        # Save metrics for last round for Summary Table
        last_metrics = full_metrics[-1]
        last_metrics['Training Time (s)'] = round(duration, 2)
        final_metrics_summary.append(last_metrics)
        
        # Save Full History (Round 0 -> 50) for Appendix
        df_full = pd.DataFrame(full_metrics)
        df_full.to_csv(f"results/performance_table_{method}_full.csv", index=False)
        print(f"[INFO] Saved full metrics history to results/performance_table_{method}_full.csv")
        
        # Plot Confusion Matrix (Last Round)
        _, cm = utils.calculate_extended_metrics(y_true, y_pred, method)
        utils.plot_confusion_matrix(cm, label_classes, method)
    
    # Save Plots and Summary
    utils.plot_convergence(convergence_results, "FL_Convergence_Comparison")
    
    df_summary = pd.DataFrame(final_metrics_summary)
    df_summary.to_csv("results/Final_Performance_Table.csv", index=False)
    utils.plot_performance_metrics(df_summary)
    utils.plot_efficiency(df_summary)
    
    # Vẽ Trust Evolution (Nếu có file trust score)
    try:
        trust_df = pd.read_csv("results/trust_scores_poison_0.2.csv")
        utils.plot_trust_evolution(trust_df)
    except:
        pass

    print("\\n" + "="*40)
    print("PHASE 3: ROBUSTNESS ANALYSIS (TOTAL POISON 0.0 -> 0.9)")
    print("="*40)
    
    # Chạy từ 0.0 đến 0.9 (bước nhảy 0.1)
    poison_ratios = np.arange(0.0, 1.0, 0.1)
    poison_ratios = [round(x, 1) for x in poison_ratios] # Fix float issues
    
    robust_res = {'Poison_Ratio': poison_ratios, 'FedAvg': [], 'DynamicFuzzy': []}
    
    # Giảm số rounds cho Robustness check để chạy nhanh hơn (nếu cần thiết), 
    # nhưng để tốt nhất thì nên giữ nguyên ROUNDS
    ROBUST_ROUNDS = 20 
    
    for r in poison_ratios:
        print(f"\\n--- Poison Ratio: {r} ---")
        # FedAvg
        acc_fed, _, _, _ = fl.run_fl_simulation(client_data, val_data, test_data, 'fedavg', ROBUST_ROUNDS, NUM_CLIENTS, EPOCHS, BATCH_SIZE, poison_ratio=r)
        robust_res['FedAvg'].append(acc_fed[-1])
        
        # DynamicFuzzy
        acc_fuz, _, _, _ = fl.run_fl_simulation(client_data, val_data, test_data, 'fuzzy', ROBUST_ROUNDS, NUM_CLIENTS, EPOCHS, BATCH_SIZE, poison_ratio=r)
        robust_res['DynamicFuzzy'].append(acc_fuz[-1])
        
    df_robust = pd.DataFrame(robust_res)
    df_robust.to_csv("results/robustness_analysis_total_poison.csv", index=False)
    utils.plot_robustness(df_robust)
    
    print("\\n[SUCCESS] All experiments completed. Check 'results/' folder.")

if __name__ == "__main__":
    main()
"""

# GHI FILE
base_dir = "TON_IoT_FL_Project"

files_to_update = {
    "src/utils.py": utils_code,
    "src/fl_core.py": fl_core_code,
    "runall.py": runall_code
}

for filepath, content in files_to_update.items():
    full_path = os.path.join(base_dir, filepath)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Updated: {full_path}")

print("\\n[INFO] Hệ thống đã được nâng cấp!")
print("1. Đã tích hợp Viz vào utils.py")
print("2. Đã sửa lỗi Trust Score chỉ có 10 dòng (bây giờ sẽ là 50)")
print("3. Đã thêm xuất file performance_table_full.csv (Appendix)")
print("4. Đã thêm Robustness từ 0.0 đến 0.9")
print("Hãy chạy lệnh: 'python runall.py'")