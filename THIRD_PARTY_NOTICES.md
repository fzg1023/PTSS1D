# Attribution and release licensing checklist

The five model implementations were extracted from the author's local experimental source tree. The new training/evaluation interface is a manuscript-aligned consolidation, not a claim of bitwise equivalence to all historical runs.

- L-WPT: adapted reproduction referring to FrusqueGaetan/Learnable-Wavelet-Transform, commit `7b9982da87ba8c4978dd217edb1e23bfa3994217`. The grouped-convolution implementation preserves filter ordering, padding and unconstrained smooth thresholds from the existing experimental reproduction.
- TEM1Dformer: local reproduction of the convolution/Transformer/residual architecture; includes the adaptive-pooling attention cap declared in the current manuscript. Cite the TEM1Dformer article in the manuscript when using this implementation. This file does not assert that the local implementation is the authors' official repository.
- 1D ViT, 1D U-Net and 1D-DnCNN: extracted from the author's existing supervised experimental implementations. Retain and verify any upstream notices before redistribution.

No license has been invented or assigned on behalf of the author. Before public redistribution, verify upstream license terms, retain required notices, verify the bibliography, and select a compatible license for the code that the author owns. Absence of a LICENSE is not a grant of reuse permission. No datasets, checkpoints or pretrained weights are distributed here.
