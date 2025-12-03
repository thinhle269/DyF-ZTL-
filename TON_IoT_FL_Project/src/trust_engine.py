import numpy as np
import pandas as pd
import torch

class TrustEvaluator:
    def __init__(self, num_clients, decay_factor=0.2, recovery_factor=0.05, alpha=0.5):
        self.num_clients = num_clients
        self.trust_scores = np.ones(num_clients) 
        self.decay = decay_factor
        self.recovery = recovery_factor
        self.alpha = alpha
        
        # Ngưỡng sàn
        self.min_safety_threshold = 0.4 
        self.current_threshold = 0.6 
        
        # Biến theo dõi vòng hiện tại để áp dụng Ân hạn
        self.current_round = 0
        
        self.history = []

    def log_round(self, round_num):
        # Cập nhật số vòng hiện tại
        self.current_round = round_num
        
        record = {'Round': round_num, 'Threshold': self.current_threshold}
        for i in range(self.num_clients):
            record[f'Client_{i}'] = self.trust_scores[i]
        self.history.append(record)

    def calculate_trust(self, local_model, global_model, val_loader, device, client_idx):
        acc = self._evaluate(local_model, val_loader, device)
        current_trust = self.trust_scores[client_idx]
        
        # --- SMART DRACONIAN LOGIC ---
        
        # Giai đoạn 1: Ân hạn (Round 0-2)
        # Vì Epoch ít, model chưa học được, Acc có thể thấp. Không phạt nặng.
        if self.current_round < 2:
            if acc < 0.1: # Chỉ phạt nếu quá tệ (như random)
                new_trust = current_trust * 0.9
            else:
                new_trust = 1.0 # Giữ nguyên hoặc hồi phục
                
        # Giai đoạn 2: Thiết quân luật (Round 3+)
        # Lúc này model tốt đã phải đạt Acc > 80-90%. Ai < 70% là Hacker.
        else:
            if acc < 0.70: 
                # Chắc chắn là Poisoning -> Phạt cực nặng
                new_trust = current_trust * self.decay 
            elif acc < 0.85:
                # Nghi ngờ
                new_trust = current_trust * 0.9
            else:
                # Tốt
                new_trust = min(1.0, current_trust + self.recovery)
            
        self.trust_scores[client_idx] = new_trust
        return new_trust, acc

    def update_dynamic_threshold(self):
        # Nếu đang ân hạn thì để ngưỡng thấp cho mọi người cùng học
        if self.current_round < 2:
            self.current_threshold = 0.1
            return 0.1

        mean_trust = np.mean(self.trust_scores)
        std_trust = np.std(self.trust_scores)
        dynamic_thresh = mean_trust - (self.alpha * std_trust)
        self.current_threshold = max(dynamic_thresh, self.min_safety_threshold)
        return self.current_threshold

    def is_trusted(self, client_idx):
        return self.trust_scores[client_idx] >= self.current_threshold

    def _evaluate(self, model, loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total > 0 else 0

    def save_history(self, filepath):
        df = pd.DataFrame(self.history)
        df.to_csv(filepath, index=False)
        print(f"[INFO] Trust scores saved to {filepath}")