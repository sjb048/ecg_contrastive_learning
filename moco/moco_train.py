import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

from sklearn.model_selection import train_test_split


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
    
def create_ecg_dataloader(ecg_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(ecg_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True )

def train_moco_1d(model, ecg_tensor, epochs=10, batch_size=32, device='cuda'):
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
            if i % 10 == 0:
                progress.display(i)
        scheduler.step()
        avg_loss = total_loss / num_steps
        print(f"[Epoch {epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), 'improved_moco_ecg_model.pth')
    print("MoCo model saved to 'improved_moco_ecg_model.pth'.")
