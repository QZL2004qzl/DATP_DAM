import os
import copy
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml  
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# 建立数据落盘目录
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BaseMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaseMLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.regressor = nn.Linear(32, 1)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        out = self.regressor(features)
        if return_features:
            return out, features
        return out

class JudgeModel(nn.Module):
    def __init__(self, input_dim):
        super(JudgeModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.net(x)

class ThoughtPointNode:
    def __init__(self, anchor_feature):
        self.anchor = anchor_feature.detach().clone()
        self.expert = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.expert.parameters(), lr=0.01, weight_decay=1e-2)
        self.usage_count = 1       
        self.health = 1.0          

    def train_on_the_fly(self, f, residual_y, steps=5):
        self.expert.train()
        criterion = nn.MSELoss()
        f_tmp = f.detach().clone().requires_grad_(False)
        res_tmp = residual_y.detach().clone().requires_grad_(False)
        for _ in range(steps):
            self.optimizer.zero_grad()
            pred = self.expert(f_tmp)
            loss = criterion(pred, res_tmp) + 0.01 * torch.sum(self.expert.weight ** 2)
            loss.backward()
            self.optimizer.step()

# ==========================================
# 2. 嵌入思维点容器
# ==========================================
class DAThoughtPointMLP(nn.Module):
    def __init__(self, base_model, judge_model, error_gate=0.55, dist_threshold=1.5, max_points=150, enable_evolution=True):
        super(DAThoughtPointMLP, self).__init__()
        self.base_model = base_model
        self.judge_model = judge_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.judge_model.parameters():
            param.requires_grad = False
            
        self.error_gate = error_gate  
        self.dist_threshold = dist_threshold  
        self.max_points = max_points
        self.enable_evolution = enable_evolution  
        self.thought_points = []

    def apply_evolution_dynamics(self):
        if not self.enable_evolution or len(self.thought_points) < 2:
            return
        
        merged_points = []
        skip_indices = set()
        
        for i in range(len(self.thought_points)):
            if i in skip_indices:
                continue
            tp_i = self.thought_points[i]
            
            for j in range(i + 1, len(self.thought_points)):
                if j in skip_indices:
                    continue
                tp_j = self.thought_points[j]
                
                dist = torch.norm(tp_i.anchor - tp_j.anchor, p=2).item()
                if dist < (self.dist_threshold * 0.4):  
                    tp_i.anchor = (tp_i.anchor + tp_j.anchor) / 2.0
                    with torch.no_grad():
                        tp_i.expert.weight.copy_((tp_i.expert.weight + tp_j.expert.weight) / 2.0)
                        tp_i.expert.bias.copy_((tp_i.expert.bias + tp_j.expert.bias) / 2.0)
                    tp_i.usage_count += tp_j.usage_count
                    tp_i.health = min(1.0, tp_i.health + 0.1)
                    skip_indices.add(j)
            
            merged_points.append(tp_i)
            
        final_points = []
        for tp in merged_points:
            if tp.usage_count == 0: 
                tp.health -= 0.15  
            else:
                tp.health = min(1.0, tp.health + 0.05)  
                
            tp.usage_count = 0  
            
            if tp.health >= 0.2:
                final_points.append(tp)
                
        self.thought_points = final_points

    def forward(self, x, y_env=None):
        self.base_model.eval()
        self.judge_model.eval()
        
        with torch.no_grad():
            base_out, features = self.base_model(x, return_features=True)
            wrong_probabilities = self.judge_model(x)
            
        final_preds = []
        
        for i in range(len(x)):
            f_sample = features[i].view(1, -1)
            b_pred = base_out[i].view(1, 1)
            prob_wrong = wrong_probabilities[i].item()
            final_pred_val = b_pred.item()
            

            if prob_wrong <= self.error_gate:
                pass
                
  
            else:
                best_tp_idx = None
                min_div = float('inf')
                if len(self.thought_points) > 0:
                    divs = [torch.norm(f_sample.squeeze() - tp.anchor, p=2).item() for tp in self.thought_points]
                    best_tp_idx = np.argmin(divs)
                    min_div = divs[best_tp_idx]
                
      
                if y_env is not None:
                    y_true = y_env[i].view(1, 1)
                    residual_target = y_true - b_pred
                    
                    if min_div <= self.dist_threshold and best_tp_idx is not None:
                        target_tp = self.thought_points[best_tp_idx]
                        with torch.enable_grad():
                            target_tp.train_on_the_fly(f_sample, residual_target, steps=2)
                        target_tp.expert.eval()
                        with torch.no_grad():
     
                            tp_res_val = np.clip(target_tp.expert(f_sample).item(), -0.5, 0.5)
                            final_pred_val = b_pred.item() + 0.2 * tp_res_val
                        target_tp.usage_count += 1
                        
                    elif len(self.thought_points) < self.max_points:
                        new_tp = ThoughtPointNode(f_sample.squeeze())
                        with torch.enable_grad():
                            new_tp.train_on_the_fly(f_sample, residual_target, steps=5)
                        new_tp.expert.eval()
                        with torch.no_grad():
                            tp_res_val = np.clip(new_tp.expert(f_sample).item(), -0.5, 0.5)
                            final_pred_val = b_pred.item() + 0.2 * tp_res_val
                        self.thought_points.append(new_tp)
                    
                    self.apply_evolution_dynamics()
                

                elif len(self.thought_points) > 0 and min_div <= self.dist_threshold:
                    self.thought_points[best_tp_idx].expert.eval()
                    with torch.no_grad():
                        tp_res = self.thought_points[best_tp_idx].expert(f_sample).item()
   
                        tp_res = np.clip(tp_res, -0.5, 0.5)  
                        final_pred_val = b_pred.item() + 0.2 * tp_res
                        self.thought_points[best_tp_idx].usage_count += 1
                        
            final_preds.append(final_pred_val)
            
        return torch.tensor(final_preds).view(-1, 1)


datasets_to_run = [
    {"name": "kin8nm", "id": 189},              
    {"name": "house_16h", "id": 574},            
    {"name": "year_prediction_msd", "id": 44040}, 
    {"name": "diamonds", "id": 42225},          
    {"name": "delays_housing", "id": 42571},
]

def run_tuned_academic_benchmark(seeds=[42, 100, 2026]):
    global_results = {}
    
    
    for ds in datasets_to_run:
        print(f"\n【数据集同步】正在获取 OpenML ID: {ds['id']} ({ds['name']})...")
        try:
            bunch = fetch_openml(data_id=ds['id'], as_frame=True, parser='auto')
            X_raw, y_raw = bunch.data, bunch.target
            
            X_df = pd.get_dummies(X_raw, drop_first=True).fillna(0).astype(np.float32)
            y_df = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(np.float32)
            
            max_samples = 3000
            if len(X_df) > max_samples:
                X_df = X_df.sample(max_samples, random_state=42)
                y_df = y_df.loc[X_df.index]
                
            X_arr = X_df.values
            y_arr = y_df.values.reshape(-1, 1)
            
            dataset_trials = []
            
            for seed in seeds:
                X_train, X_temp, y_train, y_temp = train_test_split(X_arr, y_arr, test_size=0.5, random_state=seed)
                X_evolve, X_test, y_evolve, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
                
                scaler_X = StandardScaler().fit(X_train)
                scaler_y = StandardScaler().fit(y_train)
                
                t_X_train = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32)
                t_y_train = torch.tensor(scaler_y.transform(y_train), dtype=torch.float32)
                t_X_evolve = torch.tensor(scaler_X.transform(X_evolve), dtype=torch.float32)
                t_y_evolve = torch.tensor(scaler_y.transform(y_evolve), dtype=torch.float32)
                t_X_test = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32)
                t_y_test = torch.tensor(scaler_y.transform(y_test), dtype=torch.float32)
                

                input_dim = X_arr.shape[1]
                base_model = BaseMLP(input_dim)
                judge_model = JudgeModel(input_dim)
                optimizer = optim.Adam(list(base_model.parameters()) + list(judge_model.parameters()), lr=0.01)
                loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=64, shuffle=True)
                
                base_model.train()
                judge_model.train()
                for epoch in range(12):
                    for bx, by in loader:
                        optimizer.zero_grad()
                        pred = base_model(bx)
                        loss_base = nn.MSELoss()(pred, by)
                        with torch.no_grad():
                            is_wrong = (torch.abs(pred - by) > 0.35).float()
                        pred_wrong_prob = judge_model(bx)
                        loss_judge = nn.BCELoss()(pred_wrong_prob, is_wrong)
                        (loss_base + loss_judge).backward()
                        optimizer.step()
                        
                base_model.eval()
                judge_model.eval()
                
   
                with torch.no_grad():
                    base_preds = base_model(t_X_test).numpy()
                base_r2 = r2_score(t_y_test.numpy(), base_preds)
                
   
                model_no_evo = DAThoughtPointMLP(copy.deepcopy(base_model), copy.deepcopy(judge_model), error_gate=0.55, dist_threshold=1.5, max_points=1000, enable_evolution=False)
                _ = model_no_evo(t_X_evolve, y_env=t_y_evolve)
                final_pts_no_evo = len(model_no_evo.thought_points)
                preds_no_evo = model_no_evo(t_X_test, y_env=None).numpy()
                r2_no_evo = r2_score(t_y_test.numpy(), preds_no_evo)
                
  
                model_full = DAThoughtPointMLP(copy.deepcopy(base_model), copy.deepcopy(judge_model), error_gate=0.55, dist_threshold=1.5, max_points=150, enable_evolution=True)
                t0 = time.time()
                _ = model_full(t_X_evolve, y_env=t_y_evolve)
                final_pts_full = len(model_full.thought_points)
                
                preds_full = model_full(t_X_test, y_env=None).numpy()
                latency = (time.time() - t0) / len(t_X_test) * 1000
                r2_full = r2_score(t_y_test.numpy(), preds_full)
                
                dataset_trials.append({
                    "base_r2": base_r2,
                    "no_evo_r2": r2_no_evo,
                    "full_evo_r2": r2_full,
                    "nodes_no_evo": final_pts_no_evo,
                    "nodes_full": final_pts_full,
                    "latency": latency
                })
                
            df_trial = pd.DataFrame(dataset_trials)
            global_results[ds["name"]] = {
                "Base R²": f"{df_trial['base_r2'].mean():.4f} ± {df_trial['base_r2'].std():.4f}",
                "No-Evo R²": f"{df_trial['no_evo_r2'].mean():.4f} ± {df_trial['no_evo_r2'].std():.4f}",
                "Full-Evo R²": f"{df_trial['full_evo_r2'].mean():.4f} ± {df_trial['full_evo_r2'].std():.4f}",
                "Nodes (No-Evo)": f"{df_trial['nodes_no_evo'].mean():.1f} ± {df_trial['nodes_no_evo'].std():.1f}",
                "Nodes (Full-Evo)": f"{df_trial['nodes_full'].mean():.1f} ± {df_trial['nodes_full'].std():.1f}",
                "Latency (ms)": f"{df_trial['latency'].mean():.3f} ± {df_trial['latency'].std():.3f}"
            }
            
            print(f"   📊 [{ds['name']}] 嵌入思维点后和基座模型对比 -> Base R²: {global_results[ds['name']]['Base R²']} | Full-Evo R²: {global_results[ds['name']]['Full-Evo R²']}")
            
        except Exception as e:
            print(f"   ❌ 数据集 {ds['name']} 失败，原因: {str(e)}")


    df_academic_board = pd.DataFrame(global_results).T
    df_academic_board.to_csv(f"{OUTPUT_DIR}/results.csv")
    
    print("\n" + "📜" * 15 + "数据结果" + "📜" * 15)
    print(df_academic_board.to_string())
    print("═" * 80)
    print(f"🎉 数据保存至: {OUTPUT_DIR}/results.csv")

if __name__ == "__main__":
    run_tuned_academic_benchmark(seeds=[42, 100, 2026])