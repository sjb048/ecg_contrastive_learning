# utils_umap.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import wandb
import torch
import pandas as pd
import matplotlib.cm as cm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def log_umap_projections(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
         for batch in loader:
            # Handle both (ecg,) and (ecg, lbl)
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    ecg, lbl = batch
                else:
                    ecg = batch[0]
                    lbl = torch.zeros(ecg.size(0), dtype=torch.long)  # Dummy labels for unlabeled data
            else:
                ecg = batch
                lbl = torch.zeros(ecg.size(0), dtype=torch.long)
            emb = model.encoder(ecg.to(device)).cpu()
            embeddings.append(emb)
            labels.append(lbl)
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    
    # Handle NaN/Inf
    if np.isnan(embeddings).any():
        nan_percentage = np.isnan(embeddings).mean() * 100
        wandb.log({"nan_percentage_in_embeddings": nan_percentage})
        print(f"NaN detected in {nan_percentage:.2f}% of embeddings. Replacing with defaults.")
        embeddings = np.nan_to_num(embeddings)
    
    reducer = umap.UMAP(n_components=2, metric='cosine')
    projections = reducer.fit_transform(embeddings)
    

    fig, ax = plt.subplots()
    sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=labels, palette='viridis', ax=ax)
    wandb.log({"umap_projections": wandb.Image(fig)})
    plt.close(fig)


def log_custom_umap(emb_scaled, epoch):
    """Custom UMAP visualization with clustering"""
   
    # UMAP projection
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42,
                        metric='cosine', low_memory=True).fit_transform(emb_scaled)
    
    # Clustering
    kmeans_labels = KMeans(n_clusters=5, random_state=42).fit_predict(emb_scaled)
    dbscan_labels = DBSCAN(eps=1.5, min_samples=10).fit_predict(emb_scaled)

    k_palette = sns.color_palette("hls", len(np.unique(kmeans_labels)))
    db_palette = sns.color_palette("hls", len(np.unique(dbscan_labels)))

    
    # Create DataFrame
    df = pd.DataFrame({
        "UMAP1": umap_2d[:,0], "UMAP2": umap_2d[:,1],
        "KMeans": kmeans_labels, "DBSCAN": dbscan_labels
    })
    colors = ['green','red','blue','purple','orange']

    palette = sns.color_palette("hls", len(np.unique(kmeans_labels)))
    
    # Plot KMeans
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="KMeans", palette=k_palette, s=15)
    plt.title(f"UMAP + KMeans (epoch {epoch})")
    fname_km = f"umap_kmeans_epoch_{epoch}.png"
    plt.savefig(fname_km)
    plt.close()
    wandb.log({f"UMAP_KMeans_ep{epoch}": wandb.Image(fname_km)})
    
    # Plot DBSCAN
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="DBSCAN", palette=db_palette, s=15)
    plt.title(f"UMAP + DBSCAN (epoch {epoch})")
    fname_db = f"umap_dbscan_epoch_{epoch}.png"
    plt.savefig(fname_db)
    plt.close()
    wandb.log({f"UMAP_DBSCAN_ep{epoch}": wandb.Image(fname_db)})

def log_confusion_matrix(y_true, y_pred, step_tag, show_local=False,save_path=None):
    """Draw & W&B-log a confusion-matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False, square=True,
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({step_tag})")
    wandb.log({f"cm/{step_tag}": wandb.Image(fig)})
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
        
    if show_local:
        plt.show()
    else:
        plt.close(fig)

def log_tsne_projections(emb_scaled, epoch):
    """t-SNE visualization with clustering"""
   
    
    # t-SNE projection (perplexity is an important parameter)
    tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(emb_scaled)
    
    # Clustering
    kmeans_labels = KMeans(n_clusters=5, random_state=42).fit_predict(emb_scaled)
    
    # Create DataFrame
    df = pd.DataFrame({
        "TSNE1": tsne_2d[:,0], "TSNE2": tsne_2d[:,1],
        "KMeans": kmeans_labels
    })
    
    # Plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="KMeans", palette='viridis', s=15)
    plt.title(f"t-SNE Visualization (epoch {epoch})")
    fname = f"tsne_epoch_{epoch}.png"
    plt.savefig(fname)
    plt.close()
    wandb.log({f"TSNE_ep{epoch}": wandb.Image(fname)})
