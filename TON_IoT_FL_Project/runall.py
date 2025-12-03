import src.preprocessing as prep
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
    
    # ====================================================
    # CẤU HÌNH THÍ NGHIỆM (Sửa số vòng tại đây)
    # ====================================================
    NUM_CLIENTS = 20
    ROUNDS = 100        # <-- SỬA THÀNH 100 NẾU CẦN
    EPOCHS = 5         
    BATCH_SIZE = 32
    # ====================================================
    
    print("="*50)
    print(f"BẮT ĐẦU CHẠY THÍ NGHIỆM VỚI {ROUNDS} VÒNG")
    print("="*50)
    
    print("\n[PHASE 1] DATA PREPARATION")
    client_data, val_data, test_data, num_classes, input_dim, label_classes = prep.load_and_process_data(DATA_PATH, NUM_CLIENTS)
    
    print("\n[PHASE 2] PERFORMANCE & APPENDIX GENERATION")
    
    convergence_results = {}
    final_metrics_summary = [] 
    
    methods = ['FedAvg', 'FedProx', 'DynamicFuzzy']
    
    for method in methods:
        algo_name = 'fuzzy' if method == 'DynamicFuzzy' else method.lower()
        print(f"\n>>> Running Method: {method}...")
        
        start = time.time()
        # Chạy mô phỏng
        acc_hist, y_true, y_pred, full_metrics = fl.run_fl_simulation(
            client_data, val_data, test_data, algo_name, ROUNDS, NUM_CLIENTS, EPOCHS, BATCH_SIZE
        )
        duration = time.time() - start
        
        convergence_results[method] = acc_hist
        
        # Lưu metric vòng cuối cùng cho bảng tổng hợp
        last_metrics = full_metrics[-1]
        last_metrics['Training Time (s)'] = round(duration, 2)
        final_metrics_summary.append(last_metrics)
        
        # 1. LƯU FULL HISTORY (Cho Appendix)
        df_full = pd.DataFrame(full_metrics)
        df_full.to_csv(f"results/performance_table_{method}_full.csv", index=False)
        print(f"[INFO] Saved full metrics history: results/performance_table_{method}_full.csv")
        
        # 2. VẼ CONFUSION MATRIX (Vòng cuối)
        _, cm = utils.calculate_extended_metrics(y_true, y_pred, method)
        utils.save_confusion_matrix_csv(cm, label_classes, method)
        utils.plot_confusion_matrix(cm, label_classes, method)
    
    # Vẽ biểu đồ hội tụ
    utils.plot_convergence(convergence_results, "FL_Convergence_Comparison")
    
    # Lưu và vẽ bảng tổng hợp cuối cùng
    df_summary = pd.DataFrame(final_metrics_summary)
    df_summary.to_csv("results/Final_Performance_Table.csv", index=False)
    utils.plot_performance_metrics(df_summary)
    utils.plot_efficiency(df_summary)
    
    # Vẽ Trust Evolution (nếu có file)
    try:
        trust_df = pd.read_csv("results/trust_scores_poison_0.2.csv")
        utils.plot_trust_evolution(trust_df)
    except:
        pass

    print("\n[PHASE 3] ROBUSTNESS ANALYSIS (POISON 0.0 -> 0.9)")
    
    # Chạy dải poison rộng để test độ lỳ đòn
    poison_ratios = np.arange(0.0, 1.0, 0.1)
    poison_ratios = [round(x, 1) for x in poison_ratios] 
    
    robust_res = {'Poison_Ratio': poison_ratios, 'FedAvg': [], 'DynamicFuzzy': []}
    
    # Để tiết kiệm thời gian, phần này có thể chạy ít vòng hơn (VD: 20)
    # Nhưng để đồng bộ, ta cứ dùng biến ROUNDS chung
    
    for r in poison_ratios:
        print(f"\n--- Testing Poison Ratio: {r} ---")
        # FedAvg
        acc_fed, _, _, _ = fl.run_fl_simulation(client_data, val_data, test_data, 'fedavg', 20, NUM_CLIENTS, EPOCHS, BATCH_SIZE, poison_ratio=r)
        robust_res['FedAvg'].append(acc_fed[-1])
        
        # DynamicFuzzy
        acc_fuz, _, _, _ = fl.run_fl_simulation(client_data, val_data, test_data, 'fuzzy', 20, NUM_CLIENTS, EPOCHS, BATCH_SIZE, poison_ratio=r)
        robust_res['DynamicFuzzy'].append(acc_fuz[-1])
        
    df_robust = pd.DataFrame(robust_res)
    df_robust.to_csv("results/robustness_analysis_total_poison.csv", index=False)
    utils.plot_robustness(df_robust)
    
    print("\n" + "="*50)
    print("[SUCCESS] HOÀN TẤT TOÀN BỘ THÍ NGHIỆM!")
    print("Kiểm tra thư mục 'results/' để lấy hình ảnh và file CSV.")
    print("="*50)

if __name__ == "__main__":
    main()