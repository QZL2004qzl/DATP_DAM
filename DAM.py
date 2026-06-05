import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class Config:

    experiment_name = "wikitext103_DAM_baseline"
    max_train_tokens = 50000000  
    global_batch_size = 16        
    
    # 根据你的显存大小调整（如 16 或 32）
    micro_batch_size = 16
    # 动态计算梯度累积步数，严格保证每个 Effective Step 看到的 Token 数与全局 Batch 一致
    grad_accum_steps = global_batch_size // micro_batch_size 
    
    # =========================================================================
    # 🏗️ 模型架构参数
    # =========================================================================
    vocab_size = 8000
    d_model = 1024
    n_heads = 8
    d_ff = 2048
    n_layers = 8
    max_seq_len = 1024
    dropout = 0.1
    
    # 训练常规参数
    epochs = 20  # 最大 Epochs（如果先达到 max_train_tokens 则会提前终止）
    lr = 5e-4
    grad_clip = 1.0
    
    # 路径与设备
    data_dir = './wikitext_data'
    tokenizer_path = './tokenizer_wikitext_8k.json'
    log_dir = './paper_logs'
    checkpoint_dir = './paper_checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_interval = 20    # 每隔多少个 Global Step 记录一次 JSONL
    eval_interval = 100  # 每隔多少个 Global Step 评估一次并存模型

config = Config()
config.batch_size = config.micro_batch_size 

# 创建必要目录
os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)
print(f"Using device: {config.device} | Micro Batch: {config.micro_batch_size} | Grad Accum Steps: {config.grad_accum_steps}")

def download_and_save_dataset():
    """下载并保存 WikiText-103"""
    if os.path.exists(os.path.join(config.data_dir, 'train.txt')):
        print("Dataset already exists locally.")
        return
    print("Downloading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)
    for split in ["train", "validation", "test"]:
        texts = [t for t in dataset[split]["text"] if t.strip()]
        with open(os.path.join(config.data_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
    print("Dataset saved successfully!")

download_and_save_dataset()

def load_local_dataset(split='train'):
    with open(os.path.join(config.data_dir, f'{split}.txt'), 'r', encoding='utf-8') as f:
        return [t.strip() for t in f.readlines() if t.strip()]

train_texts = load_local_dataset('train')
val_texts = load_local_dataset('validation')

def get_tokenizer():
    if os.path.exists(config.tokenizer_path):
        return Tokenizer.from_file(config.tokenizer_path)
    print("Training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"], min_frequency=2)
    tokenizer.train([os.path.join(config.data_dir, 'train.txt')], trainer)
    tokenizer.save(config.tokenizer_path)
    return tokenizer

tokenizer = get_tokenizer()
pad_token_id = tokenizer.token_to_id("<pad>")
bos_token_id = tokenizer.token_to_id("<bos>")
eos_token_id = tokenizer.token_to_id("<eos>")

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len, split='train'):
        self.split = split
        cache_file = os.path.join('./', f'wiki_cache_{split}_{config.vocab_size}.pt')
        
        if os.path.exists(cache_file):
            print(f"[{split}] Loading tokenized data from cache...")
            self.sequences = torch.load(cache_file)
        else:
            print(f"[{split}] Tokenizing texts...")
            all_ids = []
            batch_size = 10000
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_encoded = tokenizer.encode_batch([f"<bos>{t}<eos>" for t in texts[i:i+batch_size]])
                for encoded in batch_encoded:
                    all_ids.extend(encoded.ids)
            
            self.sequences = [all_ids[i:i + max_seq_len + 1] for i in range(0, len(all_ids) - max_seq_len, max_seq_len)]
            torch.save(self.sequences, cache_file)
            print(f"[{split}] Created {len(self.sequences)} sequences and cached.")
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

train_dataset = WikiTextDataset(train_texts, tokenizer, config.max_seq_len, 'train')
val_dataset = WikiTextDataset(val_texts, tokenizer, config.max_seq_len, 'val')

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def DAM(self, Q, K, V, is_causal=True, past_key_value=None, use_cache=False):
        b, h, n, d = Q.shape
        m = K.shape[2]
        phi_Q = F.elu(Q) + 1.0
        phi_K = F.elu(K) + 1.0
        eps = 1e-5
        
        if is_causal:
            if past_key_value is None:
                # Prefill 阶段
                assert n == m, "Causal mask requires..."
                kv_prod = phi_K * V                
                kv_cumsum = torch.cumsum(kv_prod, dim=2)  
                den_cumsum = torch.cumsum(phi_K, dim=2) + eps 
                raw_ratio = kv_cumsum / den_cumsum
                output = phi_Q * raw_ratio
                output = F.layer_norm(output, (d,)) 
                current_total_len = m
                if use_cache:
                    last_kv = kv_cumsum[:, :, -1:, :]     
                    last_den = den_cumsum[:, :, -1:, :]   
            else:
                # Decode 阶段
                prev_kv = past_key_value["kv_sum"]        
                prev_den = past_key_value["den_sum"]      
                past_len = past_key_value["seq_len"]
                kv_cumsum = prev_kv + (phi_K * V)  
                den_cumsum = prev_den + phi_K             
                output = phi_Q * (kv_cumsum / (den_cumsum + eps))
                output = F.layer_norm(output, (d,))
                current_total_len = past_len + m
                if use_cache:
                    last_kv = kv_cumsum
                    last_den = den_cumsum
                    
            if use_cache:
                next_past_key_value = {
                    "kv_sum": last_kv,      
                    "den_sum": last_den,    
                    "seq_len": current_total_len
                }
            else:
                next_past_key_value = None
                
        else:
            kv_sum = (phi_K * V).sum(dim=2, keepdim=True)  
            den_sum = phi_K.sum(dim=2, keepdim=True) + eps        
            output = phi_Q * (kv_sum / den_sum)
            output = F.layer_norm(output, (d,))
            next_past_key_value = None
            
        return output, next_past_key_value
        
    def forward(self, x, mask=None):
        B, S, D = x.size()
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        output, _ = self.DAM(Q, K, V)
        out = self.dropout(output).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask=None):
        # Pre-LN 残差连接
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
            
    def forward(self, x):
        B, S = x.size()
        mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        x = self.dropout(self.pos_encoding(self.token_embedding(x) * math.sqrt(config.d_model)))
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))

class PaperLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_file = os.path.join(log_dir, f"{experiment_name}.jsonl")
        
    def log(self, metrics: dict):
        metrics['timestamp'] = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')

    def print_current(self, m: dict):
        print(f"\n[MHA Transformer] Step: {m['step']} | Epoch: {m['epoch']:.2f}")
        print(f"Tokens Seen: {m['tokens_seen']:,} / {config.max_train_tokens:,}")
        print(f"Train/Val Loss: {m['train_loss']:.4f} / {m['val_loss']:.4f} | Val PPL: {m['val_perplexity']:.2f}")
        print(f"Throughput: {m['throughput_tokens_per_sec']:.1f} tokens/s | Peak VRAM: {m['peak_vram_gb']:.2f} GB")
        print("-" * 70)


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
            logits = model(batch_x)
            loss = criterion(logits.view(-1, config.vocab_size), batch_y.view(-1))
            non_pad_mask = (batch_y != pad_token_id).float()
            loss = (loss * non_pad_mask.view(-1)).sum() / non_pad_mask.sum()
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss, math.exp(min(avg_loss, 20))

def train_epoch(model, train_loader, val_loader, optimizer, scheduler, criterion, epoch, logger, global_step, tokens_seen):
    model.train()
    total_loss = 0
    step_start_time = time.time()
    step_tokens = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        if tokens_seen >= config.max_train_tokens:
            print(f"\n[Budget Reached] Total tokens processed reached target ({tokens_seen:,}). Stopping training!")
            return total_loss / (batch_idx + 1), global_step, tokens_seen, True

        batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
        non_pad_mask = (batch_y != pad_token_id).float()
        num_tokens = non_pad_mask.sum().item()
        step_tokens += num_tokens
        tokens_seen += num_tokens
        logits = model(batch_x)
        loss = criterion(logits.view(-1, config.vocab_size), batch_y.view(-1))
        loss = (loss * non_pad_mask.view(-1)).sum() / non_pad_mask.sum()
        loss = loss / config.grad_accum_steps
        loss.backward()
        total_loss += loss.item() * config.grad_accum_steps
        
        if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            if global_step % config.log_interval == 0:
                elapsed_time = time.time() - step_start_time
                throughput = step_tokens / elapsed_time if elapsed_time > 0 else 0
                
                peak_vram = torch.cuda.max_memory_allocated(config.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                avg_train_loss = total_loss / (batch_idx + 1)
                val_loss, val_ppl = evaluate(model, val_loader, criterion)
                
                metrics = {
                    "step": global_step,
                    "epoch": epoch + batch_idx / len(train_loader),
                    "tokens_seen": int(tokens_seen),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(val_loss),
                    "val_perplexity": float(val_ppl),
                    "throughput_tokens_per_sec": float(throughput),
                    "peak_vram_gb": float(peak_vram),
                    "lr": float(scheduler.get_last_lr()[0])
                }
                logger.log(metrics)
                logger.print_current(metrics)
                
                step_start_time = time.time()
                step_tokens = 0
                model.train()

            if global_step % config.eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, criterion)
                checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_MHA_step.pt')
                torch.save({'step': global_step, 'tokens_seen': tokens_seen, 'model_state_dict': model.state_dict()}, checkpoint_path)
                print(f"Checkpoint successfully saved: {checkpoint_path}")
                model.train()

        pbar.set_postfix({'loss': f'{loss.item() * config.grad_accum_steps:.4f}', 'tokens': f'{int(tokens_seen):,}'})
        
    return total_loss / len(train_loader), global_step, tokens_seen, False

def main():
    print("=" * 70)
    print(f"Starting Project: {config.experiment_name}")
    print(f"Target Compute Budget: {config.max_train_tokens:,} Tokens")
    print("=" * 70)
    
    model = Model(
        vocab_size=config.vocab_size, d_model=config.d_model, n_heads=config.n_heads,
        d_ff=config.d_ff, n_layers=config.n_layers, max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(config.device)
    

    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = sum(p.numel() for n, p in model.named_parameters() if "token_embedding" not in n and "lm_head" not in n)
    print(f"Total Parameters: {total_params:,}")
    print(f"Non-embedding Parameters (Core Layers): {non_emb_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
    logger = PaperLogger(config.log_dir, config.experiment_name)

    total_estimated_steps = (config.max_train_tokens // config.global_batch_size) // config.max_seq_len
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_estimated_steps, eta_min=1e-5)
    
    global_step = 0
    tokens_seen = 0
    stop_training = False
    
    for epoch in range(config.epochs):
        if stop_training:
            break
        print(f"\n--- Starting Epoch {epoch + 1} ---")
        _, global_step, tokens_seen, stop_training = train_epoch(
            model, train_loader, val_loader, optimizer, scheduler, criterion, epoch + 1, logger, global_step, tokens_seen
        )
        
    print(f"\n🎉 Baseline Experiment Completed! Data safely output to {logger.log_file}")

if __name__ == "__main__":
    main()