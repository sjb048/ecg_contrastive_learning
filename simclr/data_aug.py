# IMPORTS
import  os
import  random
import  numpy as np
import  pandas as pd
import pywt
import  time

import  torch
import  torch.nn as nn
from    torchvision import transforms

def wavelet_denoise(signal_1d, wavelet='db4', level=2, mode='soft'):
    coeff = pywt.wavedec(signal_1d, wavelet, mode="per")
    sigma = np.median(np.abs(coeff[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal_1d)))
    coeff[1:] = [pywt.threshold(c, value=uthresh, mode=mode) for c in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")

class RandomCrop1D(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, signal):
        # signal shape: (channels, time)
        _, t = signal.shape
        if t == self.crop_size:
            return signal
        if t < self.crop_size:
            raise ValueError("Signal length is smaller than crop_size.")
        start = random.randint(0, t - self.crop_size)
        return signal[:, start:start+self.crop_size]

class RandomGrayscale1D(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, signal):
        if random.random() < self.p:
            # If signal has more than one channel, average across channels.
            # Otherwise, return the signal as is.
            if signal.dim() > 1 and signal.size(0) > 1:
                return torch.mean(signal, dim=0, keepdim=True)
            else:
                return signal
        else:
            return signal

class RandomScaling1D(object):
    def __init__(self, scale_min=0.8, scale_max=1.2):
        self.scale_min = scale_min
        self.scale_max = scale_max
        
    def __call__(self, signal):
        factor = random.uniform(self.scale_min, self.scale_max)
        return signal * factor
    
class RandomAddNoise1D(object):
    """Adds Gaussian noise to a 1D signal."""
    def __init__(self, noise_std=0.02):
        self.noise_std = noise_std
        
    def __call__(self, signal):
        noise = torch.randn_like(signal) * self.noise_std
        return signal + noise

class RandomFlip1D(object):
    """Randomly reverses the 1D signal (if applicable)."""
    def __call__(self, signal):
        if random.random() < 0.5:
            return torch.flip(signal, dims=[-1])
        return signal

class RandomTimeShift1D(object):
    """Randomly shifts the 1D signal in time."""
    def __init__(self, max_shift=10):
        self.max_shift = max_shift  # maximum shift in samples
        
    def __call__(self, signal):
        shift = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(signal, shifts=shift, dims=-1)
        
class Normalize1D(object):
    """Normalize the 1D signal with a given mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, signal):
        return (signal - self.mean) / self.std

class RandomIntensityJitter1D(object):
    
    def __init__(self, brightness=0.8, contrast=0.8):
       
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, signal):
        # Random offset (brightness change)
        brightness_offset = random.uniform(-self.brightness, self.brightness)
        # Random scaling (contrast change)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        return (signal + brightness_offset) * contrast_factor
    
# Compose the 1D augmentations:

class augmentation_simclr(Dataset):
    def __init__(self, x):
        # We create a dataset that has multiple copies with different augmentation labels.
        self.data = [(el, 'none') for el in x]
        self.data.extend([(el, 'augm1') for el in x])
        self.data.extend([(el, 'augm2') for el in x])
        self.data.extend([(el, 'augm3') for el in x])
        
        # Instead of image-specific transforms, use the custom 1D transforms.
        self.transform1 = transforms.Compose([
            RandomCrop1D(crop_size =224),   # Crop a segment of length 224 from the signal
            RandomIntensityJitter1D(brightness=0.8, contrast=0.8),
            RandomScaling1D(scale_min = 0.8, scale_max = 1.2),  # Random scaling
            RandomAddNoise1D(noise_std=0.02),          # <-- Add noise here
            Normalize1D(mean = 0.5, std = 0.5)     # Example normalization (tweak as needed)
        ])
        self.transform2 = transforms.Compose([
            RandomFlip1D(),                # Random flip/reversal (if appropriate)
            RandomScaling1D(scale_min = 0.8, scale_max = 1.2),  # Random scaling
            RandomGrayscale1D(p=0.2),  # Applies grayscale conversion with probability 0.2
            RandomTimeShift1D(max_shift = 8),
            RandomAddNoise1D(noise_std=0.02),          # <-- Add noise here
            Normalize1D(mean = 0.5, std = 0.5)
        ])
        self.transform3 = transforms.Compose([
            RandomCrop1D(crop_size = 224),   # Crop a segment of length 224 from the signal
            RandomTimeShift1D(max_shift = 10),
            RandomAddNoise1D(noise_std=0.02),          # <-- Add noise here
            Normalize1D(mean = 0.5, std = 0.5)
        ])

    def __getitem__(self, index):
        # Here choose which augmentation to apply based on label
        signal, aug_label = self.data[index]

        # Convert signal to a tensor if it isn’t already
        if not torch.is_tensor(signal):
            signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            
        if aug_label == 'none':
            return signal
        elif aug_label == 'augm1':
            return self.transform1(signal)
        elif aug_label == 'augm2':
            return self.transform2(signal)
        elif aug_label == 'augm3':
            return self.transform3(signal)
    
    def __len__(self):
        return len(self.data)
    
