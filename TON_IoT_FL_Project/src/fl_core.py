import torch
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
                
                # FedProx Regularization
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
            
    return y_true, y_pred

def run_fl_simulation(client_datasets, val_dataset, test_dataset, model_type, rounds, num_clients, epochs, batch_size, poison_ratio=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = client_datasets[0][0][0].shape[0]
    
    all_labels = []
    for _, y in test_dataset:
        all_labels.append(y.item())
    num_classes = len(set(all_labels))

    # Khởi tạo mô hình Global
    if model_type == 'fuzzy':
        global_model = src.models.DynamicFuzzyNet(input_dim, num_classes).to(device)
    else:
        global_model = src.models.DeepNet(input_dim, num_classes).to(device)
        
    global_weights = global_model.state_dict()
    
    # Các biến lưu lịch sử
    acc_history = []
    full_metrics_history = [] 
    final_y_true, final_y_pred = [], []
    
    # Khởi tạo Trust Engine (Chỉ dùng cho Fuzzy)
    trust_engine = TrustEvaluator(num_clients, alpha=0.5)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Thiết lập Poisoning
    num_poisoned = int(num_clients * poison_ratio)
    poisoned_indices = np.random.choice(range(num_clients), num_poisoned, replace=False)
    if num_poisoned > 0:
        print(f"[WARN] Poisoning clients: {poisoned_indices}")

    for r in tqdm(range(rounds), desc=f"Training {model_type} (Poison={poison_ratio})"):
        local_weights_candidates = []
        client_indices_candidates = []
        
        idxs_users = range(num_clients)
        
        # --- BƯỚC 1: TRAIN & TÍNH TRUST ---
        for idx in idxs_users:
            if model_type == 'fuzzy':
                local_model = src.models.DynamicFuzzyNet(input_dim, num_classes).to(device)
            else:
                local_model = src.models.DeepNet(input_dim, num_classes).to(device)
                
            local_model.load_state_dict(global_weights)
            algo = 'fedprox' if model_type == 'fedprox' else 'fedavg'
            
            # Logic Tấn công (Label Flipping)
            dataset_to_use = client_datasets[idx]
            if idx in poisoned_indices:
                X_p, y_p = dataset_to_use[:][0].clone(), dataset_to_use[:][1].clone()
                y_p = (y_p + 1) % num_classes 
                dataset_to_use = torch.utils.data.TensorDataset(X_p, y_p)
            
            trainer = LocalUpdate(dataset_to_use, batch_size, 0.01, epochs, device, algo=algo)
            w, _ = trainer.train(local_model, global_model if algo == 'fedprox' else None)
            
            # Nếu là Fuzzy thì tính Trust
            if model_type == 'fuzzy':
                local_model.load_state_dict(w)
                trust_engine.calculate_trust(local_model, None, val_loader, device, idx)
                local_weights_candidates.append(w)
                client_indices_candidates.append(idx)
            else:
                local_weights_candidates.append(w)

        # --- BƯỚC 2: LỌC ZERO TRUST (Adaptive Filtering) ---
        final_local_weights = []
        if model_type == 'fuzzy':
            trust_engine.update_dynamic_threshold()
            for i, w in zip(client_indices_candidates, local_weights_candidates):
                if trust_engine.is_trusted(i):
                    final_local_weights.append(w)
            trust_engine.log_round(r) # Lưu log trust
        else:
            final_local_weights = local_weights_candidates

        # --- BƯỚC 3: TỔNG HỢP (Aggregation) ---
        if len(final_local_weights) > 0:
            global_weights = average_weights(final_local_weights)
            global_model.load_state_dict(global_weights)
        
        # --- BƯỚC 4: ĐÁNH GIÁ & LƯU METRICS CHI TIẾT ---
        y_true, y_pred = evaluate_model(global_model, test_dataset, device)
        
        round_metrics, _ = utils.calculate_extended_metrics(y_true, y_pred, model_type)
        round_metrics['Round'] = r
        full_metrics_history.append(round_metrics)
        
        acc_history.append(round_metrics['Accuracy'])
        
        if r == rounds - 1:
            final_y_true = y_true
            final_y_pred = y_pred
            
    # Lưu lịch sử Trust nếu đúng điều kiện
    if model_type == 'fuzzy' and poison_ratio == 0.2:
        trust_engine.save_history(f"results/trust_scores_poison_{poison_ratio}.csv")
        
    return acc_history, final_y_true, final_y_pred, full_metrics_history