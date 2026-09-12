"""Prepared clean .npy arrays, deterministic sampling, and paired corruption."""
import numpy as np
import torch
from torch.utils.data import Dataset


def make_pair(clean, snr_db, noise_type, rng):
    # AWGN: do NOT center or normalize each white-noise realization.
    noise = rng.standard_normal(clean.size)
    if noise_type == "pink":
        freq = np.fft.rfftfreq(clean.size)
        scale = np.ones_like(freq)
        scale[1:] = 1.0 / np.sqrt(freq[1:] + 1e-12)
        noise = np.fft.irfft(np.fft.rfft(noise) * scale, n=clean.size)
        noise = (noise - noise.mean()) / (noise.std() + 1e-12)
    elif noise_type != "white":
        raise ValueError("noise_type must be white or pink")
    power = np.mean(clean.astype(np.float64) ** 2) + 1e-12
    noisy = (clean + noise * np.sqrt(power / 10 ** (snr_db / 10))).astype(np.float32)
    mean, std = noisy.mean(), noisy.std() + 1e-8
    x, y = (clean - mean) / std, (noisy - mean) / std
    return x.astype(np.float32), y.astype(np.float32)


class Signals(Dataset):
    """Strict [N,L] or [N,1,L] input; no silent channel selection/flattening."""
    def __init__(self, path, length, noise_type, seed=42, indices=None,
                 training=False, snr=None, min_snr=-15., max_snr=50.):
        self.data = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.data.ndim not in (2, 3) or (self.data.ndim == 3 and self.data.shape[1] != 1):
            raise ValueError("Expected clean numeric array [N,L] or [N,1,L]")
        if self.data.dtype.kind not in "fiu" or self.data.shape[-1] < 2:
            raise ValueError("Expected numeric signals with at least two samples")
        self.indices = np.arange(len(self.data)) if indices is None else np.asarray(indices)
        if not len(self.indices):
            raise ValueError("Empty dataset/split")
        self.length, self.noise_type, self.seed = length, noise_type, seed
        self.training, self.snr = training, snr
        self.min_snr, self.max_snr, self.epoch = min_snr, max_snr, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        # Independent of model, target, worker count and iteration order.
        snr_key = 0 if self.snr is None else int(round((self.snr + 1000) * 1000))
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, index, self.epoch if self.training else 0, snr_key]))
        clean = np.asarray(self.data[index], dtype=np.float32).reshape(-1)
        if not np.isfinite(clean).all():
            raise ValueError("Non-finite values in sample %d" % index)
        if len(clean) > self.length:
            start = int(rng.integers(len(clean) - self.length + 1)) if self.training else 0
            clean = clean[start:start + self.length]
        elif len(clean) < self.length:
            clean = np.pad(clean, (0, self.length - len(clean)))
        snr = self.snr
        if snr is None:
            sigma = rng.uniform(10 ** (-self.max_snr / 20), 10 ** (-self.min_snr / 20))
            snr = -20 * np.log10(sigma + 1e-12)
        x, y = make_pair(clean, snr, self.noise_type, rng)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])
