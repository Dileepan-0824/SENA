import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate_and_plot(data_dir: Path, output_dir: Path):
    """
    Generate final reporting outputs including SENA feature distributions 
    and model prediction curves, saving them as images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot SENA Feature Distributions
    logging.info("Plotting SENA feature distributions...")
    try:
        diagnostics = pd.read_csv(data_dir / 'processed' / 'sena_diagnostics.csv')
        sena_features = ['in_degree', 'out_degree', 'constraint', 'effective_size', 
                         'homophily_ratio', 'temporal_betweenness']
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(sena_features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(diagnostics[feature].dropna(), bins=30, kde=True, color='skyblue')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sena_feature_distributions.png', dpi=300)
        plt.close()
        
        # Plot over time (trend)
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(sena_features, 1):
            plt.subplot(2, 3, i)
            sns.lineplot(data=diagnostics, x='bin_id', y=feature, estimator='mean', errorbar=None)
            plt.title(f'Trend of {feature} over time')
            plt.xlabel('Time Bin (Weeks)')
            plt.ylabel(f'Mean {feature}')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'sena_feature_trends.png', dpi=300)
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting features: {e}")

    # 2. Plot Prediction Metrics (ROC and PR Curves for the PyTorch Model)
    logging.info("Plotting model prediction performances...")
    try:
        targets = [('target_a', 'Target A: Rank Drop Risk'), 
                   ('target_b', 'Target B: Constraint Increase Risk')]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for idx, (t_name, title_desc) in enumerate(targets):
            pred_file = data_dir / 'processed' / f'preds_{t_name}.csv'
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                y_true = df['y_true'].values
                y_pred = df['y_pred'].values
                
                if len(np.unique(y_true)) > 1:
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_true, y_pred)
                    roc_auc = auc(fpr, tpr)
                    axes[0].plot(fpr, tpr, lw=2, label=f'{title_desc} (AUC = {roc_auc:.2f})')
                    
                    # PR Curve
                    prec, rec, _ = precision_recall_curve(y_true, y_pred)
                    pr_auc = auc(rec, prec)
                    axes[1].plot(rec, prec, lw=2, label=f'{title_desc} (AP = {pr_auc:.2f})')
                
        # Setup ROC plot
        axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Receiver Operating Characteristic')
        axes[0].legend(loc="lower right")
        
        # Setup PR plot
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(output_dir / 'custom_model_roc_pr_curves.png', dpi=300)
        plt.close()
        
        # Plot prediction probabilities distributions
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (t_name, title_desc) in enumerate(targets):
            pred_file = data_dir / 'processed' / f'preds_{t_name}.csv'
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                sns.histplot(data=df, x='y_pred', hue='y_true', bins=30, kde=True, 
                             ax=axes[idx], palette='Set2')
                axes[idx].set_title(f'Prediction Probs: {title_desc}')
                axes[idx].set_xlabel('Predicted Probability')
                
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_probability_distributions.png', dpi=300)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting predictions: {e}")

    logging.info("Visualizations successfully saved to outputs/")

if __name__ == '__main__':
    evaluate_and_plot(Path('data'), Path('outputs'))
