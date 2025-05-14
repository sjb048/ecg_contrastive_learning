import os, random, time, gc, ast, logging, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb, pywt,wandb                # <-  pywt was missing
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torch.amp import GradScaler, autocast  
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils_umap import (
    log_confusion_matrix,          # keep this – rename our own function later
    log_umap_projections,
    log_custom_umap,
    log_tsne_projections
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_rows = [] 
# At start of main pipeline
full_df = pd.read_csv('../data/ptbxl_database.csv')
train_df = full_df[full_df.strat_fold.isin([1,2,3,4,5,6,7,8])]  # Official training folds
val_df = full_df[full_df.strat_fold == 9]                       # Official validation fold
test_df = full_df[full_df.strat_fold == 10]                     # Official test fold

# --- DATA LOADING FOR UNLABELED DATA (for MoCo Pre-training) ---
def load_and_segment_signals_1d_no_folder_limit(
    csv_file,
    base_dir,
    filename_col='filename_lr', #records100 used for low resolution
    max_files=5000,
    segment_length=1000,
    fs=125.0,
    do_wavelet=False,
    wavelet='db4',
    level=2,
    mode='soft',
    lead_index=0,
    random_files=False
):
    """
    Loads up to max_files ECG signals from the CSV.
    For each record, extracts the first segment_length samples from one lead.
    Optionally applies wavelet denoising.
    Returns: a DataFrame of segments and a list of missing files.
    """
    df = pd.read_csv(csv_file)
    if random_files:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)


    df = df.iloc[:max_files].copy()

    loaded_segments = []
    missing_files = []

    for _, row in df.iterrows():
        rel_path = row[filename_col]
        full_hea_path = os.path.join(base_dir, rel_path) + '.hea'
        if not os.path.exists(full_hea_path):
            missing_files.append(full_hea_path)
            continue
        try:
            record_path_no_ext = full_hea_path[:-4]
            record = wfdb.rdrecord(record_path_no_ext)
            ecg_signal = record.p_signal  # shape: (num_samples, num_leads)
            if ecg_signal.shape[0] < segment_length:
                missing_files.append(full_hea_path)
                continue
            segment_2d = ecg_signal[:segment_length, :]
            segment_1d = segment_2d[:, lead_index]
            if do_wavelet:
                segment_1d = wavelet_denoise(segment_1d, wavelet=wavelet, level=level, mode=mode)
            seg_df = pd.DataFrame({
                "ecg_1d": segment_1d,
                "record_id": [os.path.basename(record_path_no_ext)] * segment_length
            })
            loaded_segments.append(seg_df)
        except Exception as e:
            missing_files.append(full_hea_path)
            continue
    all_segments_df = pd.concat(loaded_segments, ignore_index=True) if loaded_segments else pd.DataFrame()
    return all_segments_df, missing_files

def wavelet_denoise(ecg_signal_1d, wavelet='db4', level=2, mode='soft'):
    coeffs = pywt.wavedec(ecg_signal_1d, wavelet, level=level)
    detail_coeffs = coeffs[1:]
    detail_array = np.concatenate([np.abs(dc) for dc in detail_coeffs]) if detail_coeffs else np.array([])
    if len(detail_array) == 0:
        return ecg_signal_1d
    median_abs = np.median(detail_array)
    sigma = median_abs / 0.6745
    n = len(ecg_signal_1d)
    threshold = sigma * np.sqrt(2 * np.log(n)) if n > 1 else 0
    new_coeffs = [coeffs[0]]
    for dc in coeffs[1:]:
        dc_t = pywt.threshold(dc, threshold, mode=mode)
        new_coeffs.append(dc_t)
    denoised = pywt.waverec(new_coeffs, wavelet)
    if len(denoised) > n:
        denoised = denoised[:n]
    elif len(denoised) < n:
        denoised = np.pad(denoised, (0, n - len(denoised)), mode='constant')
    return denoised

# --- DATA AUGMENTATION FUNCTION ---
def augment_ecg(ecg_batch, crop_size=1000, noise_std=0.01):
    """
    Applies random cropping and Gaussian noise to a batch of ECG signals.
    ecg_batch shape: [batch, in_channels, seq_len]
    """
    batch_size, in_channels, seq_len = ecg_batch.shape
    augmented = ecg_batch.clone()
    for i in range(batch_size):
        # Random crop
        if seq_len > crop_size:
            start = random.randint(0, seq_len - crop_size)
            augmented[i, :, :crop_size] = augmented[i, :, start:start + crop_size]
            if crop_size < seq_len:
                augmented[i, :, crop_size:] = 0.0

       # Multi-channel noise
        actual_len = min(crop_size, seq_len)
        if noise_std > 0:
            # Generate noise for all channels
            noise = torch.randn(in_channels, actual_len, device=ecg_batch.device) * noise_std
            augmented[i, :, :actual_len] += noise
    return augmented

# --- SUPERVISED CLASSIFIER USING PRE-TRAINED ENCODER ---
class ECGClassifier(nn.Module):
    def __init__(self, encoder, emb_dim=128, num_classes=2):
        """
        Uses the pre-trained encoder to extract features from ECG,
        then applies a classification head.
        """
        super(ECGClassifier, self).__init__()
        self.encoder = encoder
        # Optionally freeze encoder parameters:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits
    
class ImprovedECGEncoder(nn.Module):
    """
    A 1D CNN encoder for ECG signals.
    Uses 4 stacked convolutional blocks and a final FC layer to produce an embedding.
    """
    def __init__(self, in_channels=1, base_channels=64, embedding_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            self._make_conv_block(in_channels, base_channels),           # => base_channels
            self._make_conv_block(base_channels, base_channels * 2),       # => base_channels*2
            self._make_conv_block(base_channels * 2, base_channels * 4),   # => base_channels*4
            self._make_conv_block(base_channels * 4, base_channels * 8)    # => base_channels*8
        )
        self.fc = nn.Linear(base_channels * 8, embedding_dim)
    def _make_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x
# --- MOCO TRAINING FUNCTIONS ---
class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
 
# MoCo1D: MoCo adapted for 1D signals
class MoCo1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, emb_dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = ImprovedECGEncoder(in_channels=in_channels, base_channels=base_channels, embedding_dim=emb_dim)
        self.encoder_k = ImprovedECGEncoder(in_channels=in_channels, base_channels=base_channels, embedding_dim=emb_dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer("queue", torch.randn(emb_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()

    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# def train_moco_1d(model, ecg_tensor, epochs, batch_size, device='cuda'):
def train_moco_1d(model, ecg_tensor, epochs, batch_size, device='cuda',
                   viz_interval=20, viz_methods=None):
    if viz_methods is None:
        viz_methods = ['umap']

    dataloader = create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.encoder_q.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    for epoch in range(epochs):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))
        end = time.time()
        total_loss = 0.0
        num_steps = 0
        for i, (batch_data,) in enumerate(dataloader):
            if isinstance(batch_data, (tuple, list)):
              batch_data = batch_data[0]

            data_time.update(time.time() - end)
            batch_data = batch_data.to(device)
            im_q = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)
            im_k = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            losses.update(loss.item(), batch_data.size(0))
            top1.update(acc1[0], batch_data.size(0))
            top5.update(acc5[0], batch_data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1
            batch_time.update(time.time() - end)
            end = time.time()
            log_rows.append({
                "epoch": epoch + 1,
                "batch": i,
                "time": batch_time.val,
                "data": data_time.val,
                "loss": losses.val,
                "acc1": top1.val.item(),
                "acc5": top5.val.item()
            })

            if i % 10 == 0:
                progress.display(i)
        scheduler.step()
        avg_loss = total_loss / num_steps
        wandb.log({"epoch": epoch+1, "moco_loss": avg_loss, "moco_acc@1": top1.avg, "moco_acc@5": top5.avg})
        print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
        print(viz_methods);
        # print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
        # Show t-SNE plot every 20 epochs
        if (epoch + 1) % viz_interval == 0:
            visualize_embeddings(model, ecg_tensor, epoch+1, batch_size, device, viz_methods)
    
    log_path = f"moco_logs_100records_epoch_{epoch+1}.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Batch-wise metrics saved to: {log_path}")
    torch.save(model.state_dict(), 'improved_moco_ecg_model_test.pth')
    print("MoCo model saved to 'improved_moco_ecg_model_test.pth'.")

def evaluate_moco(model, ecg_tensor, batch_size=64, device='cuda'):
    model.eval()
    dataloader = create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    with torch.no_grad():
        for (batch_data,) in dataloader:
            batch_data = batch_data.to(device)
            emb = model.encoder_q(batch_data)
            all_embeddings.append(emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def visualize_embeddings(model, ecg_tensor, epoch, batch_size, device, viz_methods):
    
    model.eval()
    # Extract embeddings
    embeddings = evaluate_moco(model, ecg_tensor, batch_size, device)
    
    # Convert to numpy
    emb_np = embeddings.numpy()
    
    # Check for NaNs - critical before visualization
    if np.isnan(emb_np).any():
        print(f"NaN detected in embeddings at epoch {epoch}! Skipping visualization.")
        return
    
    # Filter zero-variance dimensions with fallback
    var = emb_np.var(axis=0)
    if np.any(var > 0):
        emb_filtered = emb_np[:, var > 0]
    else:
        emb_filtered = emb_np + np.random.normal(0, 1e-6, emb_np.shape)
    
    # Standardize using RobustScaler (handles outliers better than StandardScaler)
    emb_scaled = RobustScaler().fit_transform(emb_filtered)
    
    try:
        for method in viz_methods:
            if method.lower() == 'umap':
                # Call existing utility function if it exists
                log_umap_projections(model, 
                                     create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=False), 
                                     device)
                
                # Optional: Also log the custom UMAP visualization if desired
                log_custom_umap(emb_scaled, epoch)
                
            elif method.lower() == 'tsne':
                log_tsne_projections(emb_scaled, epoch)
                
            # elif method.lower() == 'pca':
            #     log_pca_projections(emb_scaled, epoch)
                
    except Exception as e:
        print(f"Error in embedding visualization: {str(e)}")
    # Add garbage collection after intensive memory operations
    gc.collect()
    torch.cuda.empty_cache()
    print("train will be called")
    # Switch back to training mode
    model.train()


def create_ecg_dataloader(ecg_tensor, batch_size=64, shuffle=True):
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True )

# --- LABELED DATASET FOR SUPERISED FINETUNING ---
class PTBXLDataset(Dataset):
    def __init__(self, csv_path, root_dir, use_lr=True, folds=None, 
                 is_unlabeled=False,
                 transform=None):

        self.root_dir = root_dir
        self.use_lr = use_lr
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.is_unlabeled = is_unlabeled
        if folds is not None:
            if 'strat_fold' not in self.df.columns:
                raise ValueError("CSV missing 'strat_fold' column")
            self.df = self.df[self.df.strat_fold.isin(folds)]
        # Ensure the column exists in the CSV.
        if not self.is_unlabeled:  # Only process labels for supervised data
            self._process_labels()
        self._validate_columns()
        self._filter_missing_files()
       
    def _process_labels(self):
        """Only called for labeled data"""
        self.df['label'] = self.df.scp_codes.apply(
            lambda s: 0 if "NORM" in ast.literal_eval(s) else 1
        )
    def _validate_columns(self):
        required_cols = {'filename_lr', 'filename_hr', 'scp_codes', 'strat_fold'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        # data_dir = 'records100'
        data_dir = 'records100' if self.use_lr else 'records500'
        print (data_dir)
        full_path = os.path.join(self.root_dir, data_dir)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"ECG data folder not found: {full_path}")

    def _filter_missing_files(self):
        invalid_rows = []
        filename_col = 'filename_lr' if self.use_lr else 'filename_hr'
        print (filename_col)

        for idx, row in self.df.iterrows():
            wfdb_path = os.path.join(self.root_dir, row[filename_col])
            primary_hea = wfdb_path + '.hea'
            primary_dat = wfdb_path + '.dat'
            alt_path = wfdb_path.replace('_lr', '').replace('_hr', '')
            alt_hea = alt_path + '.hea'
            alt_dat = alt_path + '.dat'
            if not (os.path.exists(primary_hea) and os.path.exists(primary_dat)) and \
               not (os.path.exists(alt_hea) and os.path.exists(alt_dat)):
                invalid_rows.append(idx)

        if invalid_rows:
            self.df.drop(index=invalid_rows, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"Filtered dataset: {len(self.df)} valid records remain.")
        else:
            logger.info(f"No missing files found. {len(self.df)} valid records remain.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        filename_col = 'filename_lr'
        # filename_col = 'filename_lr' if self.use_lr else 'filename_hr'
        wfdb_path = os.path.join(self.root_dir, record[filename_col])
        try:
            signals, _ = wfdb.rdsamp(wfdb_path)
        except FileNotFoundError:
            alt_path = wfdb_path.replace('_lr', '').replace('_hr', '')
            try:
                signals, _ = wfdb.rdsamp(alt_path)
            except FileNotFoundError as e:
                logger.error(f"File not found: {wfdb_path} and alternative {alt_path}")
                raise e
        ecg = signals.T.astype(np.float32)  # (12, time)
        # For classification, we flatten the data (or you can keep it 2D if you design a multi-channel model)
        ecg_flat = ecg.reshape(-1)
        ecg_flat = np.expand_dims(ecg_flat, axis=0)  # (1, 12*time)
        if self.transform:
            ecg_flat = self.transform(ecg_flat)
        return (
            torch.tensor(ecg_flat, dtype=torch.float32),
            torch.tensor(record['label'], dtype=torch.long)
        )


def train_classifier_with_comparison(
                model,
                train_loader,
                val_loader,
                device="cpu",
                epochs=60,
                lr=1e-4,
                eval_every=20, 
                metric_for_lr="f1_score",
                
                min_delta=1e-3,
                lr_factor=0.5,
                improvement_epsilon=0.0  # Set a margin if needed
            ):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_metric = -float("inf")     
    step_tag   = lambda ep: f"ep{ep:03d}"
    model.to(device)

    current_val_accuracy = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for ecg_batch, labels_batch in train_loader:
            ecg_batch = ecg_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            # Use memory-efficient forward/backward
            with torch.cuda.amp.autocast():
                outputs = model(ecg_batch)
                loss = criterion(outputs, labels_batch)

           # Gradient accumulation friendly backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

           

            train_loss += loss.item() * ecg_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += labels_batch.size(0)

             # Memory cleanup
            del ecg_batch, labels_batch, outputs
            torch.cuda.empty_cache()

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        # print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}")
          # Log training metrics for this epoch
        wandb.log({"classifier_epoch": epoch+1, "train_loss": train_epoch_loss, "train_acc": train_epoch_acc})

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}")

        # Validate at the designated epoch
       
        if (epoch+1) % eval_every == 0:
            model.eval()
            val_metrics = evaluate_metrics(model, val_loader, device=device)
            current_val_accuracy = val_metrics['accuracy']
            wandb.log({
                "val_acc_20": val_metrics['accuracy'],
                "val_prec_20": val_metrics['precision'],
                "val_recall_20": val_metrics['recall'],
                "val_f1_20": val_metrics['f1_score'],
                "classifier_epoch": epoch + 1
            })
            log_confusion_matrix(val_metrics["ground_truth"], 
                               val_metrics["predictions"],
                               step_tag=step_tag(epoch+1),
                               show_local=False)
            
            # adaptive learning rate if validation metric does not improve
            current = val_metrics[metric_for_lr]
            if current < best_metric + min_delta:
                # no meaningful improvement → shrink LR
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_factor  # FIXED: use lr_factor parameter
                print(f"↳ metric did not improve ≥ {min_delta:.4g}; "
                    f"reducing LR by {lr_factor} → {optimizer.param_groups[0]['lr']:.2e}")
            else:
                best_metric = current
           
    return best_metric


def official_evaluation(
    encoder, 
    labeled_csv, 
    labeled_root, 
    device,
    test_fold=[10],  # Official test fold
    # num_folds=5,
    batch_size=64,
    epochs=60,
    eval_every=20
):
   
    """Optimized 5-fold cross-validation with PTB-XL best practices."""
    # Official PTB-XL splits
    TRAIN_FOLDS = list(range(1, 9))  # Folds 1-8
    VAL_FOLD = [9]
    TEST_FOLD = test_fold

   
    # =================================================================
    # 1. Prepare Official Splits
    # =================================================================
    # Training data (folds 1-8)
    # Create full dataset once
    train_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=TRAIN_FOLDS
    )
    # Validation data (fold 9)
    val_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=[9]  # Official validation fold
    )
    # Test data (fold 10)
    test_ds = PTBXLDataset(
        csv_path=labeled_csv,
        root_dir=labeled_root,
        use_lr=True,
        folds=TEST_FOLD
    )
    # =================================================================
    # 2. Create DataLoaders with Optimal Parameters
    # =================================================================
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2
    )

    # After dataset creation
    wandb.log({
        "train_class_dist": wandb.Histogram(train_ds.df['label']),
        "test_class_dist": wandb.Histogram(test_ds.df['label'])
    })

     # =================================================================
    # 3. Train Classifier with Validation Monitoring
    # =================================================================
    classifier = ECGClassifier(encoder, emb_dim=128, num_classes=2).to(device)

    best_metric = train_classifier_with_comparison(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=1e-4,
        eval_every=eval_every
    )

    # =================================================================
    # 4. Final Evaluation on Official Test Set
    # =================================================================
    classifier.eval()
    test_metrics = evaluate_metrics(classifier, test_loader, device)
    
    # Cleanup GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    return test_metrics
    
def evaluate_metrics(model, dataloader, device='cpu'):
    """
    Evaluates the given model on the provided dataloader and computes
    accuracy, precision, recall, and F1-score.
    Returns a dictionary with the metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ecg_batch, labels_batch in dataloader:
            ecg_batch = ecg_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(ecg_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'predictions': all_preds,
        'ground_truth': all_labels
    }




#
# --- MAIN PIPELINE ---
if __name__ == "__main__":
    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method("spawn", force=True)

    # Phase 1: MoCo Pre-training
    wandb.init(project="ECG-MoCo-finalumap_100", name="pretrain")
    # Load unlabeled data using official training folds
    unlabeled_ds = PTBXLDataset(
        csv_path='../data/ptbxl_database.csv',
        root_dir='../data',
        use_lr=True,
        folds=list(range(1,9)),  # Official unlabeled folds
        is_unlabeled=True
    )
    # print("is_unlabeled for unlabeled_ds:", unlabeled_ds.is_unlabeled)
    unlabeled_df, skipped = load_and_segment_signals_1d_no_folder_limit(
       
        csv_file='../data/ptbxl_database.csv',
        base_dir='../data',
        filename_col='filename_lr',
        max_files=5000,
        segment_length=1000,
        fs=125.0,
        do_wavelet=True,
        wavelet='db4',
        level=2,
        mode='soft',
        lead_index=0,
        random_files=False
    )
    print("Loaded segments shape:", unlabeled_df.shape)
    print("Number of skipped files:", len(skipped))
     # Group by record_id and convert each segment to shape (1, 1000)
    all_ecgs   = np.stack([g["ecg_1d"].values for _, g in unlabeled_df.groupby("record_id")])
    ecg_tensor = torch.from_numpy(all_ecgs[:, None, :]).float().to(device)
    print("ECG tensor shape:", ecg_tensor.shape)
    moco_model = MoCo1D(in_channels=1, base_channels=64, emb_dim=128)
  
    train_moco_1d(
        model = moco_model,
        ecg_tensor = ecg_tensor,
        epochs = 100,
        batch_size = 64,
        viz_methods = ['umap'],
        device = device
    )
    # ... training code ...
    wandb.finish()

    # Phase 2: Fine-tuning 
    # Official supervised splits
    wandb.init(
        project="ECG-MoCo-finalumap_100",
        name="finetune",
        config={
            "phase": "supervised_fine_tuning",
            "epochs": 100,
            "lr": 1e-4,
            "batch_size": 64,
            "eval_every": 20
        }
    )

    train_ds = PTBXLDataset(
        csv_path='../data/ptbxl_database.csv',
        root_dir='../data',
        use_lr=True,
        folds=list(range(1,9)),  # Official training folds
        is_unlabeled=False  # Default behavior
    )
    val_ds = PTBXLDataset(
        csv_path='../data/ptbxl_database.csv',
        root_dir='../data',
        use_lr=True,
        folds=[9]  # Official validation fold
    )
        # Create DataLoaders with proper parameters
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    # Initialize classifier with pre-trained MoCo encoder
    classifier = ECGClassifier(
        encoder=moco_model.encoder_q,
        emb_dim=128,
        num_classes=2
    ).to(device)

    # Train with validation-based monitoring
    best_f1 = train_classifier_with_comparison(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=60,
        lr=1e-4,
        eval_every=20
    )

    # Log final metrics
    wandb.log({"best_validation_f1": best_f1})
    wandb.finish()

    # Cleanup resources
    gc.collect()
    torch.cuda.empty_cache()
   
    

    # Phase 3: Evaluation
    wandb.init(project="ECG-MoCo-finalumap_100", name="evaluation")
    final_metrics = official_evaluation(
        encoder=moco_model.encoder_q,
        labeled_csv='../data/ptbxl_database.csv',
        labeled_root='../data',
        device=device,
        test_fold=[10],  # Official test fold
        batch_size=64,
        epochs=60
    )
    
    wandb.log(final_metrics)
    # ... eval code ...
    wandb.finish()