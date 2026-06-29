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

import scipy.stats
import sklearn.metrics
import sklearn.model_selection
import pandas as pd
import numpy as np


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        
        # MLP for structure embeddings (s1, s2)
        if config.use_s:
            self.structure_mlp = nn.Sequential(
                nn.Linear(config.dim_s, config.dim_s * 2),
                nn.ReLU(),
                nn.BatchNorm1d(config.dim_s * 2, momentum=0.1),
                nn.Linear(config.dim_s * 2, 1536)
            )
            dim_s_processed = 1536
        else:
            dim_s_processed = 0

        # MLP for perturbed gene expressions (p1, p2)
        if config.use_p:
            self.pgex_mlp = nn.Sequential(
                nn.Linear(config.dim_p, config.dim_p * 2),
                nn.ReLU(),
                nn.BatchNorm1d(config.dim_p * 2, momentum=0.1),
                nn.Linear(config.dim_p * 2, 1536)
            )
            dim_p_processed = 1536
        else:
            dim_p_processed = 0

        # MLP for baseline gene expressions (g)
        if config.use_g:
            self.gex_mlp = nn.Sequential(
                nn.Linear(config.dim_g, config.dim_g * 2),
                nn.ReLU(),
                nn.BatchNorm1d(config.dim_g * 2, momentum=0.1),
                nn.Linear(config.dim_g * 2, 1536)
            )
            dim_g_processed = 1536
        else:
            dim_g_processed = 0
                              
        # Calculate total input dimension for fusion layer
        # Order: t, s, p, c, g
        input_dim = 0
        if config.use_t:
            input_dim += config.dim_t
        if config.use_s:
            input_dim += dim_s_processed
        if config.use_p:
            input_dim += dim_p_processed
        if config.use_c:
            input_dim += config.dim_c
        if config.use_g:
            input_dim += dim_g_processed
        
        self.fusion_net = nn.Sequential(
            nn.BatchNorm1d(input_dim, momentum=0.1),
            nn.Linear(input_dim, config.layer0),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.BatchNorm1d(config.layer0, momentum=0.1),
            nn.Linear(config.layer0, config.layer1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.BatchNorm1d(config.layer1, momentum=0.1),
            nn.Linear(config.layer1, 1)
        )

    def forward(self, x):
        """
        Input x shape: [batch_size, total_dim]
        Order in x: t, s1, s2, p1, p2, c, g
        where t = avg(drug1_text, drug2_text)
        """
        features = []
        idx = 0
        
        if self.config.use_t:
            t = x[:, idx:idx + self.config.dim_t]
            idx += self.config.dim_t
            features.append(t)
        
        if self.config.use_s:
            s1 = x[:, idx:idx + self.config.dim_s]
            s2 = x[:, idx + self.config.dim_s:idx + 2 * self.config.dim_s]
            idx += 2 * self.config.dim_s
            
            s1_processed = self.structure_mlp(s1)
            s2_processed = self.structure_mlp(s2)
            s_avg = (s1_processed + s2_processed) * 0.5
            features.append(s_avg)

        if self.config.use_p:
            p1 = x[:, idx:idx + self.config.dim_p]
            p2 = x[:, idx + self.config.dim_p:idx + 2 * self.config.dim_p]
            idx += 2 * self.config.dim_p
            
            p1_processed = self.pgex_mlp(p1)
            p2_processed = self.pgex_mlp(p2)
            p_avg = (p1_processed + p2_processed) * 0.5
            features.append(p_avg)
        
        if self.config.use_c:
            c = x[:, idx:idx + self.config.dim_c]
            idx += self.config.dim_c
            features.append(c)
        
        if self.config.use_g:
            g = x[:, idx:idx + self.config.dim_g]
            g_processed = self.gex_mlp(g)
            idx += self.config.dim_g
            features.append(g_processed)
        
        x_fused = torch.cat(features, dim=1)
        
        return self.fusion_net(x_fused)

    def predict(self, x):
        return self.forward(x)
    

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

    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x.view(x.size(0), -1))
        return z, y

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.config.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=max(int(0.01 * self.config.epoch), 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def normalize_schema(df, config):
    df = df.copy()

    # Scale synergy values (GDSC2 only: *100)
    if config.data == "GDSC2":
        df['synergy'] = df['synergy'] * 100

    # Create key columns for structure / baseline-gex lookups
    if config.data == "DrugComb":
        df['drug_row_struct'] = df['drug_row_clean']
        df['drug_col_struct'] = df['drug_col_clean']
    else:  # O'Neil / GDSC2
        df['drug_row_struct'] = df['drug_row']
        df['drug_col_struct'] = df['drug_col']

    return df


def prepare_data(df, config, drug_embeds=None, cellline_embeds=None, 
                 structure_embeds=None, p_embeds=None, gene_embeds=None):
    train_list = []
    y_list = []

    for item in df.index:
        row = df.loc[item]
        features = []

        # t: averaged drug text embeddings
        if config.use_t:
            v1 = np.asarray(drug_embeds[row['drug_row']], dtype=float)
            v2 = np.asarray(drug_embeds[row['drug_col']], dtype=float)
            t = (v1 + v2) * 0.5
            features.append(t)
        
        # s1, s2: drug structure ECFP
        if config.use_s:
            s1 = np.asarray(structure_embeds[row['drug_row_struct']], dtype=float)
            s2 = np.asarray(structure_embeds[row['drug_col_struct']], dtype=float)
            features.extend([s1, s2])

        # p1, p2: perturbed gene expression
        if config.use_p:
            p1 = np.asarray(p_embeds[row['combi_row']], dtype=float)
            p2 = np.asarray(p_embeds[row['combi_col']], dtype=float)
            features.extend([p1, p2])
        
        # c: cell line text embedding
        if config.use_c:
            c = np.asarray(cellline_embeds[row['cell_line_name']], dtype=float)
            features.append(c)
        
        # g: baseline gene expression
        if config.use_g:
            g = np.asarray(gene_embeds[row['cell_line_name']], dtype=float)
            features.append(g)
        
        train_list.append(np.hstack(features))
        y_list.append(row['synergy'])
    
    return np.array(train_list), np.array(y_list)

    
class Config:
    def __init__(self, args, dim_t, dim_s, dim_p, dim_c, dim_g, seed):
        # Copy all args attributes
        for key, value in vars(args).items():
            setattr(self, key, value)

        # Fixed dimensions
        self.use_t = args.t
        self.use_s = args.s
        self.use_p = args.p
        self.use_c = args.c
        self.use_g = args.g
        
        # Fixed dimensions
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.dim_p = dim_p
        self.dim_c = dim_c
        self.dim_g = dim_g
        
        self.patience_es = max(int(0.1 * self.epoch), 1)
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Feature string for naming
        self.features_str = self._get_features_str()
        
    def _get_features_str(self):
        """Generate feature string based on enabled features"""
        features = ""
        if self.use_t:
            features += "t"
        if self.use_s:
            features += "s"
        if self.use_p:
            features += "p"
        if self.use_c:
            features += "c"
        if self.use_g:
            features += "g"
        return features


def main():
    parser = argparse.ArgumentParser(description='Feature Fusion for Drug Synergy Prediction')
    parser.add_argument('--data', type=str, default='ONeil', help='ONeil or DrugComb or GDSC2')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=1000, help='Max epoch')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--layer0', type=int, default=8192, help='First hidden layer size')
    parser.add_argument('--layer1', type=int, default=4096, help='Second hidden layer size')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')

    # Feature flags
    parser.add_argument('--t', action='store_true', help='Use Text (LLM Drug Embedding)')
    parser.add_argument('--s', action='store_true', help='Use Structure (ECFP)')
    parser.add_argument('--p', action='store_true', help='Use Perturbed Gene Expression')
    parser.add_argument('--c', action='store_true', help='Use Cell line Text Embedding')
    parser.add_argument('--g', action='store_true', help='Use Baseline Gene Expression')

    parser.add_argument('--gpu', type=int, default=0, help='GPU card number')
    parser.add_argument('--seed', type=int, default=2025, help='Seed number for reproducibility')
    args = parser.parse_args()

    SEED = args.seed
    L.seed_everything(SEED, workers=True)
    
    # Fixed dimensions for each modality
    DIM_T = 1536  # LLM drug text embedding
    DIM_S = 2048 # Drug structure (ECFP)
    DIM_P = 922 # Perturbed gene expression
    DIM_C = 1536  # LLM cell line text embedding
    DIM_G = 954 # Baseline gene expression

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    config = Config(args, DIM_T, DIM_S, DIM_P, DIM_C, DIM_G, SEED)

    # Validate that the requested features are available for this dataset 
    VALID_FEATURES = {
        "ONeil":    {"t", "c"},
        "GDSC2":    {"t", "s", "c", "g"},
        "DrugComb": {"t", "s", "p", "c"},
    }

    enabled = set(config.features_str)
    if not enabled:
        raise ValueError("No features enabled. Use at least one of --t/--s/--p/--c/--g")

    allowed = VALID_FEATURES[config.data]
    if not enabled <= allowed:
        raise ValueError(
            f"Features {sorted(enabled - allowed)} not supported for "
            f"'{config.data}'. Allowed: {sorted(allowed)}"
        )

    baseDir = Path(__file__).resolve().parent
    dataDir = baseDir.parent / "data"
    dataDirMap = {
        "ONeil": dataDir / "ONeil",
        "DrugComb": dataDir / "DrugComb",
        "GDSC2": dataDir / "GDSC2",
    }
    dataRoot = dataDirMap[config.data]
        
    print("=" * 80)
    print(f"Data: {config.data}")
    print(f"Using features: {config.features_str}")
    print("=" * 80)

    data_file = f"{config.data}_synergy.csv"
    embed_cell = f"{config.data}_embed_cell.pkl"
    embed_drug = f"{config.data}_embed_drug.pkl"

    if config.data == "DrugComb":
        structure_file = f"{config.data}_ecfp.pkl"
        p_file = "DrugComb_pgex.pkl"

    elif config.data == "GDSC2":
        structure_file = f"{config.data}_ecfp.pkl"
        gene_file = f"{config.data}_gex.pkl"

    # Load ground truth data
    print(f"\nLoading data from {data_file}...")
    df_groundtruth = pd.read_csv(dataRoot / data_file)
    df_groundtruth = normalize_schema(df_groundtruth, config)
    print(f"Loaded {len(df_groundtruth)} samples")

    drug_embeds = None
    if config.use_t:
        print(f"Loading drug text embeddings...")
        with open(dataRoot / embed_drug, 'rb') as f:
            drug_embeds = pickle.load(f)
    
    structure_embeds = None
    if config.use_s:
        print(f"Loading drug structure embeddings...")
        with open(dataRoot / structure_file, 'rb') as f:
            structure_embeds = pickle.load(f)

    p_embeds = None
    if config.use_p:
        print(f"Loading perturbed gene expression embeddings...")
        with open(dataRoot / p_file, 'rb') as f:
            p_embeds = pickle.load(f)

    cellline_embeds = None
    if config.use_c:
        print(f"Loading cell line text embeddings...")
        with open(dataRoot / embed_cell, 'rb') as f:
            cellline_embeds = pickle.load(f)
        
    gene_embeds = None
    if config.use_g:
        print(f"Loading baseline gene expression data...")
        with open(dataRoot / gene_file, 'rb') as f:
            gene_embeds = pickle.load(f)
    
    fold_metrics = []
    all_stopped_epochs = []
    
    for fold_idx in range(5):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1} / 5")
        print(f"{'='*80}")
        
        df_trval = df_groundtruth[df_groundtruth['fold'] != fold_idx]
        df_test = df_groundtruth[df_groundtruth['fold'] == fold_idx]
        
        X_train, y_train = prepare_data(df_trval, config, drug_embeds, cellline_embeds, 
                                        structure_embeds, p_embeds, gene_embeds)
        X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(
            X_train, y_train, random_state=config.seed
        )
        X_test, y_test = prepare_data(df_test, config, drug_embeds, cellline_embeds, 
                                        structure_embeds, p_embeds, gene_embeds)
        
        X_tr = torch.FloatTensor(X_tr)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        y_tr = torch.FloatTensor(y_tr)
        y_val = torch.FloatTensor(y_val)
        y_test = torch.FloatTensor(y_test)
        
        train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch, 
            num_workers=5, 
            shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=config.batch, num_workers=5)
        
        model = LitAutoEncoder(Encoder(config), config)

        es = EarlyStopping(
            monitor="val_loss",
            patience=config.patience_es,
            verbose=False,
            mode="min"
        )
        
        trainer = L.Trainer(
            max_epochs=config.epoch,
            callbacks=[es],
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        trainer.fit(model, train_loader, val_loader)
        
        # Check stopping epoch
        actual_stopped_epoch = 0
        if es.stopped_epoch > 0:
            actual_stopped_epoch = es.stopped_epoch
            print(f"  Early stopping triggered at epoch {actual_stopped_epoch}")
        else:
            actual_stopped_epoch = trainer.current_epoch
            print(f"  Training completed {actual_stopped_epoch} epochs")
        all_stopped_epochs.append(actual_stopped_epoch)
        
        trainer.test(model, test_loader)
        
        model.encoder.eval()
        with torch.no_grad():
            y_pred = model.encoder.predict(X_test).detach().cpu().numpy()
        
        y_true = y_test.detach().cpu().numpy().ravel()
        y_hat = y_pred.ravel()
        
        print("\nINFERENCE RESULTS")
        
        mse = sklearn.metrics.mean_squared_error(y_true, y_hat)
        cor, _ = scipy.stats.pearsonr(y_true, y_hat)
        metrics_con = {'mse': mse, 'pcor': cor}
        print(f"  MSE={mse:.4f}, PCOR={cor:.4f}")        
            
        fold_metric = {
            'fold': fold_idx + 1,
            **metrics_con,
        }
        fold_metrics.append(fold_metric)        
        fold_metrics[-1]['stopped_epoch'] = actual_stopped_epoch
    
    avg_stopped_epoch = np.mean(all_stopped_epochs)
    std_stopped_epoch = np.std(all_stopped_epochs)
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    metric_cols = [col for col in fold_metrics_df.columns if col not in ['fold', 'stopped_epoch']]
    macro_avg_metrics = fold_metrics_df[metric_cols].mean()
    macro_std_metrics = fold_metrics_df[metric_cols].std()
    
    macro_avg_metrics['stopped_epoch'] = avg_stopped_epoch
    macro_std_metrics['stopped_epoch'] = std_stopped_epoch
    
    print("\n--- Model Performance (Mean ± Std) ---")
    for metric in metric_cols:
        print(f"  {metric}: {macro_avg_metrics[metric]:.4f} ± {macro_std_metrics[metric]:.4f}")

    print(f"\nTRAINING FINISHED!")


if __name__ == "__main__":
    main()