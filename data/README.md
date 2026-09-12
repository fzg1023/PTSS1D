# Prepared clean data only

Obtain the four datasets under their respective access and licensing terms. Prepare independent source-separated training/test splits before exporting numeric .npy arrays [N,L] or [N,1,L]. This repository does not distribute data or prescribe a new official dataset split.

Expected relative files:

- audio/train.npy and audio/test.npy — length 16384
- seismic/train.npy and seismic/test.npy — length 6144
- ecg/train.npy and ecg/test.npy — length 4096
- vibration/train.npy and vibration/test.npy — length 1024

Resampling, source-level split selection and original database parsing must be performed in your own data preparation pipeline. See the root README for cropping, validation split and normalization behavior. No data files should be committed.
