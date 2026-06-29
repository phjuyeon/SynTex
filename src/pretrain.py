#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import pickle

import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import math
import scipy.stats
import sklearn.metrics
import sklearn.model_selection
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
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x.view(x.size(0), -1))
        loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x.view(x.size(0), -1))
        val_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x.view(x.size(0), -1))
        test_loss = F.mse_loss(z, y.view(x.size(0), -1))
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        patience = 3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def prepare_data(df, drug_embeds, cellline_embeds, structure_embeds, p_embeds):
    """
    Constructs raw feature vector.
    Order: [T1, S1, P1, T2, S2, P2, C]
    """
    data_list = []
    y_list = []
    
    for item in df.index:
        d1, d2, cl = df.loc[item]['drug_row'], df.loc[item]['drug_col'], df.loc[item]['cell_line_name']
        d1_structure, d2_structure = df.loc[item]['drug_row_clean'], df.loc[item]['drug_col_clean']
        combi1, combi2 = df.loc[item]['combi_row'], df.loc[item]['combi_col']
        y = df.loc[item]['synergy']
        
        features = []
        
        # --- Drug 1 Block ---
        features.append(np.asarray(drug_embeds[d1], dtype=float))
        features.append(np.asarray(structure_embeds[d1_structure], dtype=float))
        features.append(np.asarray(p_embeds[combi1], dtype=float))
            
        # --- Drug 2 Block ---
        features.append(np.asarray(drug_embeds[d2], dtype=float))
        features.append(np.asarray(structure_embeds[d2_structure], dtype=float))
        features.append(np.asarray(p_embeds[combi2], dtype=float))
            
        # --- Cell Block ---
        features.append(np.asarray(cellline_embeds[cl], dtype=float))
            
        data_list.append(np.hstack(features))
        y_list.append(y)
        
    return np.array(data_list), np.array(y_list)
    
    
class Config:
    def __init__(self, args, dim_t, dim_s, dim_p, dim_c, seed):
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # Fixed dimensions
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_p = dim_p
        self.dim_c = dim_c
        
        # Computed values
        self.patience_es = 20
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--layer0', type=int, default=8192)
    parser.add_argument('--layer1', type=int, default=4096)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=3072)    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)    
    args = parser.parse_args()  

    # Dimensions
    DIM_T = 1536
    DIM_S = 2048
    DIM_P = 922
    DIM_C = 1536

    SEED = args.seed
    L.seed_everything(SEED, workers=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config = Config(args, DIM_T, DIM_S, DIM_P, DIM_C, SEED)

    baseDir = Path(__file__).resolve().parent
    dataDir = baseDir.parent / "data" / "DrugComb"
    ckptDir = baseDir.parent / "ckpt"
    os.makedirs(ckptDir, exist_ok=True)

    data_file = "DrugComb_synergy.csv"
    df_groundtruth = pd.read_csv(dataDir / data_file)
    print(f"Loaded from: {data_file}")
    
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    drug_embeds = load_pkl(dataDir / "DrugComb_embed_drug.pkl")
    cellline_embeds = load_pkl(dataDir / "DrugComb_embed_cell.pkl")
    structure_embeds = load_pkl(dataDir / "DrugComb_ecfp.pkl")
    p_embeds = load_pkl(dataDir / "DrugComb_pgex.pkl")
        
    X, y = prepare_data(
        df_groundtruth, drug_embeds, cellline_embeds, 
        structure_embeds, p_embeds
    )

    train_idx, val_idx = sklearn.model_selection.train_test_split(
        np.arange(len(X)), test_size=0.1, random_state=config.seed
    )
    
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    print(f"Train set size: {len(X_tr)}, Val set size: {len(X_val)}")

    X_tr = torch.FloatTensor(X_tr)
    X_val = torch.FloatTensor(X_val)
    y_tr = torch.FloatTensor(y_tr)
    y_val = torch.FloatTensor(y_val)
    
    train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch, num_workers=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, num_workers=5)
    
    model = LitAutoEncoder(Encoder(config), config)

    es = EarlyStopping(
        monitor="val_loss",
        patience=config.patience_es,
        verbose=False,
        mode="min"
    )

    # Save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckptDir,
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
        auto_insert_metric_name=True
    )

    # Setup trainer
    callbacks = [es, checkpoint_callback]    
    trainer = L.Trainer(
        max_epochs=config.epoch,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )

    print("\n===== Starting Training =====")
    trainer.fit(model, train_loader, val_loader)

    print("\n===== Loading Best Model for Testing =====")
    if checkpoint_callback.best_model_path:
        print(f"Best model path: {checkpoint_callback.best_model_path}")
        best_model = LitAutoEncoder.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            encoder=Encoder(config),
            config=config
        )
    else:
        print("Warning: No best model checkpoint found, using current model")
        best_model = model
    
    print("\n===== Testing Best Model on Validation Set =====")
    trainer.test(best_model, val_loader)
    best_model.to(config.device)
    
    best_model.encoder.eval()
    with torch.no_grad():
        y_pred = best_model.encoder.predict(X_val.to(config.device)).detach().cpu().numpy()
    
    y_true = y_val.detach().cpu().numpy().ravel()
    y_hat = y_pred.ravel()
    
    mse = sklearn.metrics.mean_squared_error(y_true, y_hat)
    cor, _ = scipy.stats.pearsonr(y_true, y_hat)
    print(f"  MSE={mse:.4f}, PCOR={cor:.4f}")
    print(f"\nTRAINING FINISHED!")


if __name__ == "__main__":
    main()