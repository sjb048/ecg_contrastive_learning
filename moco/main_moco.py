
# IMPORTS
import os
import random
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pywt
import wfdb

# WAVELET DENOISING
def wavelet_denoise(ecg_signal_1d, wavelet='db4', level=2, mode='soft'):
    """
    Perform wavelet-based denoising on a 1D ECG signal using PyWavelets.
    """
    coeffs = pywt.wavedec(ecg_signal_1d, wavelet, level=level)
    detail_coeffs = coeffs[1:]

    # Estimate noise & threshold
    detail_array = np.concatenate([np.abs(dc) for dc in detail_coeffs]) if detail_coeffs else np.array([])
    if len(detail_array) == 0:
        return ecg_signal_1d  # no detail -> skip
    median_abs = np.median(detail_array)
    sigma = median_abs / 0.6745
    n = len(ecg_signal_1d)
    threshold = sigma * np.sqrt(2 * np.log(n)) if n > 1 else 0

    # Threshold detail coefficients
    new_coeffs = [coeffs[0]]  # keep approx
    for dc in coeffs[1:]:
        dc_t = pywt.threshold(dc, threshold, mode=mode)
        new_coeffs.append(dc_t)

    denoised = pywt.waverec(new_coeffs, wavelet) #Reconstruct signal
    if len(denoised) > n:
        denoised = denoised[:n]
    elif len(denoised) < n:
        denoised = np.pad(denoised, (0, n - len(denoised)), mode='constant')

    return denoised


# DATA LOADING & PREPROCESS
def load_and_segment_signals_1d_no_folder_limit(
    csv_file,
    base_dir,
    filename_col='filename_hr',
    max_files=300,
    segment_length=1000,
    fs=125.0,
    do_wavelet=False,
    wavelet='db4',
    level = 2,
    mode = 'soft',
    lead_index = 0
):
    """
    Loads up to `max_files` ECG signals from the CSV, ignoring folder uniqueness.
    Each record -> extracts first `segment_length` samples (1D), optional wavelet denoising.
    Returns: all_segments_df (DataFrame), missing_files (list).
    """

    df = pd.read_csv(csv_file)
    df = df.sample(frac = 1, random_state = 42).reset_index(drop=True)

    df = df.iloc[:max_files].copy()

    loaded_segments = []
    missing_files = []

    for _, row in df.iterrows():
        rel_path = row[filename_col]  # "records500/21000/21837_hr"
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

            # Take the first segment_length
            segment_2d = ecg_signal[:segment_length, :]
            # Convert 2D -> 1D by picking one lead
            segment_1d = segment_2d[:, lead_index]

            # (Optional) wavelet denoising
            if do_wavelet:
                segment_1d = wavelet_denoise(
                    ecg_signal_1d=segment_1d,
                    wavelet=wavelet,
                    level=level,
                    mode=mode
                )

            # Build a small DataFrame
            seg_df = pd.DataFrame({
                "ecg_1d": segment_1d,
                "record_id": [os.path.basename(record_path_no_ext)] * segment_length
            })
            loaded_segments.append(seg_df)

        except Exception as e:
            missing_files.append(full_hea_path)
            continue

    all_segments_df = (
        pd.concat(loaded_segments, ignore_index=True) if loaded_segments else pd.DataFrame()
    )
    return all_segments_df, missing_files


# 1D ENCODER CLASS
class ImprovedECGEncoder(nn.Module):
    """
    A more advanced 1D CNN encoder for ECG with 4 stacked Conv blocks.
    Each block: Conv1d -> BatchNorm1d -> ReLU
    Then a final Global Pool + FC => embedding_dim
    """
    def __init__(self, in_channels=1, base_channels=64, embedding_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            self._make_conv_block(in_channels, base_channels),           # => base_channels
            self._make_conv_block(base_channels, base_channels * 2),     # => base_channels*2
            self._make_conv_block(base_channels * 2, base_channels * 4), # => base_channels*4
            self._make_conv_block(base_channels * 4, base_channels * 8)  # => base_channels*8
        )

        self.fc = nn.Linear(base_channels * 8, embedding_dim)

    def _make_conv_block(self, in_c, out_c):
        # Larger kernel_size=15, stride=2, padding=7
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: [batch, in_channels, seq_len]
        x = self.conv_layers(x)  # => [batch, base_channels*8, seq_len/(2^4)]
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)  # => [batch, embedding_dim]
        x = F.normalize(x, dim=1)
        return x

#   MOCO 1D CLASS
class MoCo1D(nn.Module):
    """
    Momentum Contrast (MoCo) adapted for 1D signals.
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 emb_dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # Build query & key encoders with improved architecture
        self.encoder_q = ImprovedECGEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            embedding_dim=emb_dim
        )
        self.encoder_k = ImprovedECGEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            embedding_dim=emb_dim
        )

        # Match weights initially
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create memory bank (queue)
        self.register_buffer("queue", torch.randn(emb_dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # Momentum update of key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # keys: [batch_size, emb_dim]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K #move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        im_q: [batch, in_channels, seq_len]
        im_k: [batch, in_channels, seq_len]
        Returns (logits, labels).
        """
        # Query features
        q = self.encoder_q(im_q)  # => [batch, emb_dim]
        q = F.normalize(q, dim=1)

        # Key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)  # => [batch, emb_dim]
            k = F.normalize(k, dim=1)

        # Positive logits: q dot k
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # => [batch, 1]

        # Negative logits: q dot all in self.queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # => [batch, K]

        # Concat
        logits = torch.cat([l_pos, l_neg], dim=1)  # => [batch, 1 + K]
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels


#        AUGMENTATION FUNCTION
def augment_ecg(ecg_batch, crop_size=1000, noise_std=0.01):
    """
    ecg_batch: shape [batch, in_channels, seq_len]
    Returns a new tensor with random crop & noise.
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

        # Add noise
        actual_len = min(crop_size, seq_len)
        if noise_std > 0:
            noise = torch.randn(actual_len, device=ecg_batch.device) * noise_std
            # Single-lead => channel=0
            augmented[i, 0, :actual_len] += noise

    return augmented


#         DATA LOADER & TRAINING LOOP
def create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=True):
    """
    ecg_tensor: [N, 1, 1000]
    """
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#helper function for training moco 1d
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def train_moco_1d(model, ecg_tensor, epochs=10, batch_size=32, device='cuda'):
    """
    Trains the MoCo1D model on ecg_tensor for a given # of epochs.
    """
    dataloader = create_ecg_dataloader(ecg_tensor, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.encoder_q.parameters(), lr = 1e-4)
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

        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )
        end = time.time();
        total_loss = 0.0
        num_steps = 0

        for i, (batch_data,) in enumerate(dataloader):
            data_time.update(time.time() - end)

            batch_data = batch_data.to(device)

            # Two augmented views
            im_q = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)
            im_k = augment_ecg(batch_data, crop_size=1000, noise_std=0.01)

            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            losses.update(loss.item(), batch_data.size(0))
            top1.update(acc1[0], batch_data.size(0))
            top5.update(acc5[0], batch_data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_steps += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)


        scheduler.step()
        avg_loss = total_loss / num_steps
        print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'improved_moco_ecg_model.pth')
    print("Model saved to 'improved_moco_ecg_model.pth'.")



def evaluate_moco(model, ecg_tensor, batch_size=32, device='cuda'):
    """
    Evaluate (extract embeddings) from the model's query encoder.
    """
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


#            EXAMPLE MAIN CODE
if __name__ == "__main__":
    # CSV & base paths
    csv_file = 'train_unlabeled.csv'
    base_dir = '/content/drive/MyDrive/Masters_Thesis/unzip_data/ptbxl'

    # Step A: Load
    all_segments_1d_df, skipped = load_and_segment_signals_1d_no_folder_limit(
        csv_file=csv_file,
        base_dir=base_dir,
        filename_col='filename_hr',
        max_files=300,
        segment_length=1000,
        fs=125.0,
        do_wavelet=True,
        wavelet='db4',
        level=2,
        mode='soft',
        lead_index=0
    )
    print("Loaded shape:", all_segments_1d_df.shape)
    print("Skipped:", len(skipped))

    # Group by record_id => convert to [N, 1, 1000]
    all_ecgs = []
    record_ids = []
    for rid, group_df in all_segments_1d_df.groupby('record_id'):
        ecg_1d = group_df["ecg_1d"].values  # (1000,)
        all_ecgs.append(ecg_1d)
        record_ids.append(rid)
    all_ecgs = np.array(all_ecgs)  # => shape (N, 1000)
    all_ecgs = all_ecgs[:, np.newaxis, :]  # => (N, 1, 1000)

    ecg_tensor = torch.from_numpy(all_ecgs).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_tensor = ecg_tensor.to(device)

    print("ecg_tensor shape:", ecg_tensor.shape)
    print("Number of records loaded:", len(record_ids))

    # Step B: Instantiate MoCo1D with ImprovedECGEncoder
    moco_model = MoCo1D(
        in_channels=1,
        base_channels=64,
        emb_dim=128,
        K=65536,
        m=0.999,
        T=0.07
    )

    # Step C: Train for 5 epochs
    train_moco_1d(
        model=moco_model,
        ecg_tensor=ecg_tensor,
        epochs=5,
        batch_size=32,
        device=device
    )

    # Step D: Evaluate => get final embeddings
    embeddings = evaluate_moco(moco_model, ecg_tensor, device=device)
    print("Final embeddings shape:", embeddings.shape)

    # Optional: pick one record to plot
    import matplotlib.pyplot as plt

    example_record_id = record_ids[0]
    example_ecg_1d = all_segments_1d_df.loc[
        all_segments_1d_df['record_id'] == example_record_id, 'ecg_1d'
    ].values  # => (1000,)

    # (Optional) wavelet-denoise again if you want to see the difference
    example_ecg_denoised = wavelet_denoise(example_ecg_1d, wavelet='db4', level=2, mode='soft')

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(example_ecg_1d, label='Raw')
    plt.title(f"Raw ECG - record_id = {example_record_id}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(example_ecg_denoised, color='orange', label='Denoised')
    plt.title("Wavelet Denoised ECG")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()
