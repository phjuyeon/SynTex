#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import pickle
import yaml
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torch import nn

import math
import scipy.stats
import sklearn.metrics
import pandas as pd
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)        
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        V_context = torch.matmul(attn_weights, V)
        
        V_context = V_context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(V_context)

        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x, attn_weights = self.self_attn(x, mask=mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        return x, attn_weights


class ModalityProjector(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim * 2),
            nn.Linear(input_dim * 2, target_dim)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = 1536
        
        # --- Projectors ---
        self.s_proj = ModalityProjector(config.dim_s, self.d_model)
        self.p_proj = ModalityProjector(config.dim_p, self.d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=config.nhead,
                d_ff=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        flattened_dim = self.d_model * 2
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(flattened_dim, momentum=0.1),
            nn.Linear(flattened_dim, config.layer0),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.BatchNorm1d(config.layer0, momentum=0.1),
            nn.Linear(config.layer0, config.layer1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.BatchNorm1d(config.layer1, momentum=0.1),
            nn.Linear(config.layer1, 1)
        )
        
    def forward(self, x, return_attn=False):
        idx = 0
        seq_parts = []
        
        # --- 1. Drug Features Parsing ---
        t1, s1, p1 = None, None, None
        t2, s2, p2 = None, None, None
        
        # Drug 1
        t1 = x[:, idx:idx + self.config.dim_t]
        idx += self.config.dim_t

        s1 = x[:, idx:idx + self.config.dim_s]
        idx += self.config.dim_s

        p1 = x[:, idx:idx + self.config.dim_p]
        idx += self.config.dim_p
            
        # Drug 2
        t2 = x[:, idx:idx + self.config.dim_t]
        idx += self.config.dim_t

        s2 = x[:, idx:idx + self.config.dim_s]
        idx += self.config.dim_s

        p2 = x[:, idx:idx + self.config.dim_p]
        idx += self.config.dim_p
            
        # Cell context
        c = x[:, idx:idx + self.config.dim_c]
        idx += self.config.dim_c

        # --- 2. Sequence Construction ---        
        # Token 0: Global Anchor (Average Text)
        t_avg = (t1 + t2) * 0.5
        seq_parts.append(t_avg.unsqueeze(1))
            
        # Drug 1 Tokens
        seq_parts.append(self.s_proj(s1).unsqueeze(1))
        seq_parts.append(self.p_proj(p1).unsqueeze(1))
        
        # Drug 2 Tokens
        seq_parts.append(self.s_proj(s2).unsqueeze(1))
        seq_parts.append(self.p_proj(p2).unsqueeze(1))
        
        h = torch.cat(seq_parts, dim=1)
        
        # --- 3. Transformer Encoding ---
        final_attn_weights = []
        for layer in self.layers:
            h, attn_weights = layer(h)
            final_attn_weights = attn_weights
            
        # --- 4. Prediction ---
        pooled = h[:, 0, :] # get the first token (drut text emb avg) only
        pooled = torch.cat([pooled, c] , dim=1)
        
        output = self.head(pooled)
        
        if return_attn:
            return output, final_attn_weights

        return output

    def predict(self, x, return_attn=False):
        return self.forward(x, return_attn=return_attn)
    

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        
    def forward(self, x):
        return self.encoder(x)


def prepare_data(df, drug_embeds, cellline_embeds, structure_embeds, p_embeds):
    data_list = []
    y_list = []
    
    for item in df.index:
        d1, d2, cl = df.loc[item]['drug_row'], df.loc[item]['drug_col'], df.loc[item]['cell_line_name']
        combi1, combi2 = d1, d2
        y = df.loc[item]['synergy']
        
        features = []
        
        # --- Drug 1 Block ---
        features.append(np.asarray(drug_embeds[d1], dtype=float))
        features.append(np.asarray(structure_embeds[d1], dtype=float))
        features.append(np.asarray(p_embeds[combi1], dtype=float))
            
        # --- Drug 2 Block ---
        features.append(np.asarray(drug_embeds[d2], dtype=float))
        features.append(np.asarray(structure_embeds[d2], dtype=float))
        features.append(np.asarray(p_embeds[combi2], dtype=float))
            
        # --- Cell Block ---
        features.append(np.asarray(cellline_embeds[cl], dtype=float))
            
        data_list.append(np.hstack(features))
        y_list.append(y)
        
    return np.array(data_list), np.array(y_list)


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Inference for Drug Synergy Prediction')
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to run on: 'cpu' (default) or 'cuda', or 'cuda:0' etc.")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    baseDir = Path(__file__).resolve().parent
    dataDir = baseDir.parent / "data" / "UTSW"
    ckptDir = baseDir.parent / "_ckpt"
    
    config_path = baseDir / "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        pretrain_config_dict = yaml.safe_load(f)

    # Create config object from config of pretraining
    config = SimpleNamespace(**pretrain_config_dict)
    config.device = device

    data_file = "UTSW_synergy.csv"
    df_groundtruth = pd.read_csv(dataDir / data_file)
     
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    drug_embeds = load_pkl(dataDir / "UTSW_embed_drug.pkl")
    cellline_embeds = load_pkl(dataDir / "UTSW_embed_cell.pkl")
    structure_embeds = load_pkl(dataDir / "UTSW_ecfp.pkl")
    p_embeds = load_pkl(dataDir / "UTSW_pgex.pkl")
    
    X_test_, y_test_ = prepare_data(
        df_groundtruth, 
        drug_embeds, cellline_embeds,
        structure_embeds, p_embeds
    )
    
    X_test = torch.FloatTensor(X_test_)
    y_test = torch.FloatTensor(y_test_)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, num_workers=0, shuffle=False)
        
    state_dict = torch.load(
        ckptDir / "model.pt",
        map_location=device
    )
    model = LitAutoEncoder(Encoder(config), config).to(device)
    model.load_state_dict(state_dict)
    
    X_test = X_test.to(device)
    model.encoder.eval()
    with torch.no_grad():
        y_pred = model.encoder.predict(X_test).detach().cpu().numpy()
    
    y_true = y_test.numpy().ravel()
    y_pred = y_pred.ravel()
    
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    cor, _ = scipy.stats.pearsonr(y_true, y_pred)
    
    print(f"MSE: {mse:.2f}")
    print(f"Pearson Correlation: {cor:.2f}")
    

if __name__ == "__main__":
    main()