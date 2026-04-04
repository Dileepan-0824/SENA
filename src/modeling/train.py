import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.modeling.custom_model import train_pytorch_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate(data_dir: Path):
    """
    Phase 7: Custom Modeling Pipeline
    Train target models with strict forward temporal splits.
    Phase 8: Robustness and reporting.
    """
    modeling_data = pd.read_csv(data_dir / 'processed' / 'modeling_targets.csv')
    
    # 7.1 Build feature matrix (use local base attributes and SENA temporal metrics)
    features = ['in_degree', 'out_degree', 'constraint', 'effective_size', 
                'homophily_ratio', 'temporal_betweenness']
                
    X = modeling_data[features]
    y_a = modeling_data['target_a']
    y_b = modeling_data['target_b']
    bins = modeling_data['bin_id']
    
    # 7.2 custom risk model (e.g., Random Forest adapted for temporal constraints)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # 7.3 sanity baselines (Logistic regression)
    lr_baseline = LogisticRegression(random_state=42)
    
    # 7.4 Evaluate with strict forward temporal splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    def eval_baseline(model, X, y, cv):
        aucs, aps = [], []
        all_preds = []
        all_y = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            if len(np.unique(y_train)) > 1:
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, preds))
                aps.append(average_precision_score(y_test, preds))
                all_preds.extend(preds)
                all_y.extend(y_test)
        return np.mean(aucs) if aucs else 0, np.mean(aps) if aps else 0, np.array(all_preds), np.array(all_y)

    def eval_pytorch(X, y, cv):
        aucs, aps = [], []
        all_preds = []
        all_y = []
        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            if len(np.unique(y_train)) > 1:
                # Normalizing values for NN stability
                X_tr_mean = X_train.mean()
                X_tr_std = X_train.std().replace(0, 1)
                X_train_norm = (X_train - X_tr_mean) / X_tr_std
                X_test_norm = (X_test - X_tr_mean) / X_tr_std
                
                preds = train_pytorch_model(X_train_norm, y_train, X_test_norm, y_test)
                # Catch empty test cases effectively
                if len(np.unique(y_test)) > 1:
                    aucs.append(roc_auc_score(y_test, preds))
                aps.append(average_precision_score(y_test, preds))
                all_preds.extend(preds)
                all_y.extend(y_test)
        return np.mean(aucs) if aucs else 0, np.mean(aps) if aps else 0, np.array(all_preds), np.array(all_y)

    logging.info("Training PyTorch & Baseline models for Target A...")
    pt_auc_a, pt_ap_a, pt_preds_a, pt_y_a = eval_pytorch(X, y_a, tscv)
    lr_auc_a, lr_ap_a, _, _ = eval_baseline(lr_baseline, X, y_a, tscv)
    
    logging.info("Training PyTorch & Baseline models for Target B...")
    pt_auc_b, pt_ap_b, pt_preds_b, pt_y_b = eval_pytorch(X, y_b, tscv)
    lr_auc_b, lr_ap_b, _, _ = eval_baseline(lr_baseline, X, y_b, tscv)
    
    # Save predictions
    pd.DataFrame({'y_true': pt_y_a, 'y_pred': pt_preds_a}).to_csv(data_dir / 'processed' / 'preds_target_a.csv', index=False)
    pd.DataFrame({'y_true': pt_y_b, 'y_pred': pt_preds_b}).to_csv(data_dir / 'processed' / 'preds_target_b.csv', index=False)
    
    # 7.5 Report performance
    logging.info(f"Target A (Rank Drop)   - PyTorch Model AUC: {pt_auc_a:.3f} | Baseline AUC: {lr_auc_a:.3f}")
    logging.info(f"Target B (Constraint)  - PyTorch Model AUC: {pt_auc_b:.3f} | Baseline AUC: {lr_auc_b:.3f}")
    
    # Feature importances for final narrative
    rf_model.fit(X.fillna(0), y_a)
    fi = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
    logging.info("\nFeature Importances for Target A:")
    logging.info(f"\n{fi.to_string(index=False)}")
    
    # 8.1 - 8.4 Further ablations and report narratives will read off this logic.
    fi.to_csv(data_dir / 'processed' / 'feature_importances.csv', index=False)
    
if __name__ == '__main__':
    train_and_evaluate(Path('data'))
