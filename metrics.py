"""Per-signal metrics in the shared noisy-normalized signal space."""
import numpy as np


def metrics(clean, noisy, prediction):
    x, y, z = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (clean, noisy, prediction))
    energy = np.sum(x*x) + 1e-12
    snr_in = 10 * np.log10(energy / (np.sum((x-y)**2) + 1e-12))
    snr_out = 10 * np.log10(energy / (np.sum((x-z)**2) + 1e-12))
    a, b = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(z))
    a, b = a-a.mean(), b-b.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    # Pearson correlation is undefined for constant amplitude spectra.
    fpcc = float(a@b/denom) if denom > 1e-12 else float('nan')
    return dict(input_snr_actual_db=snr_in, output_snr_db=snr_out,
                snr_improvement_db=snr_out-snr_in,
                rmse=np.sqrt(np.mean((x-z)**2)), fpcc=fpcc)
