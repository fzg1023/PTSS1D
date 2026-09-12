"""Model-only configuration. No data loading, noise generation or training."""
from copy import deepcopy

from .vit1d import UnconditionalJiT1D
from .unet1d import PureUNet1D
from .dncnn1d import DnCNN1D
from .lwpt import LearnableWaveletPacketDenoiser
from .tem1dformer import TEM1Dformer

DATASET_LENGTHS = {"Audio": 16384, "Seismic": 6144, "ECG": 4096, "Vibration": 1024}

# Native constructor arguments; changing these changes the instantiated model.
MODEL_CONFIGS = {
    "1D-ViT": dict(signal_length=1024, patch_size=32, in_channels=1,
                   hidden_size=128, depth=4, num_heads=8),
    "1D-U-Net": dict(in_ch=1, base_ch=32, stages=4, conv_per_block=2,
                     kernel_size=3, norm_type="gn", act_type="gelu"),
    "1D-DnCNN": dict(in_channels=1, num_features=32, num_layers=18),
    "L-WPT": dict(levels=5),
    "TEM1Dformer": dict(channels=32, num_heads=4, mlp_ratio=2,
                        num_residual_blocks=10, max_tokens=256),
}

MODEL_CLASSES = {
    "1D-ViT": UnconditionalJiT1D,
    "1D-U-Net": PureUNet1D,
    "1D-DnCNN": DnCNN1D,
    "L-WPT": LearnableWaveletPacketDenoiser,
    "TEM1Dformer": TEM1Dformer,
}

# Manuscript-aligned paired architecture: identical activation for all targets.
UNET_ACTIVATIONS = {
    ("white", "x-pred"): "gelu",
    ("white", "n-pred"): "gelu",
    ("pink", "x-pred"): "gelu",
    ("pink", "n-pred"): "gelu",
}

def get_config(model_name, **overrides):
    """Return an independent configuration dictionary with checked overrides."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError("Unknown model: " + str(model_name))
    config = deepcopy(MODEL_CONFIGS[model_name])
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError("Unsupported parameters for %s: %s" % (model_name, sorted(unknown)))
    config.update(overrides)
    return config

def build_model(model_name, **overrides):
    """Build a raw network; original state_dict names are preserved."""
    config = get_config(model_name, **overrides)
    for key, value in config.items():
        if isinstance(MODEL_CONFIGS[model_name][key], int):
            if type(value) is not int or value < 1:
                raise ValueError(key + " must be a positive integer")
    if model_name == "1D-ViT":
        if config["signal_length"] % config["patch_size"]:
            raise ValueError("signal_length must be divisible by patch_size")
        if config["hidden_size"] % (2 * config["num_heads"]):
            raise ValueError("hidden_size / num_heads must be an even integer for RoPE")
    if model_name == "1D-U-Net" and config["kernel_size"] % 2 == 0:
        raise ValueError("U-Net requires an odd kernel_size to preserve length")
    if model_name == "1D-DnCNN" and config["num_layers"] < 2:
        raise ValueError("num_layers must be at least 2")
    return MODEL_CLASSES[model_name](**config)

def vit_config(dataset, configuration=3):
    """Paper configurations #1--#5; #3 is shared by both capacity sweeps."""
    if dataset not in DATASET_LENGTHS:
        raise ValueError("dataset must be one of " + str(tuple(DATASET_LENGTHS)))
    settings = {1: (128, 128), 2: (64, 128), 3: (32, 128),
                4: (32, 512), 5: (32, 256)}
    if configuration not in settings:
        raise ValueError("configuration must be 1, 2, 3, 4 or 5")
    tokens, hidden = settings[configuration]
    length = DATASET_LENGTHS[dataset]
    return get_config("1D-ViT", signal_length=length,
                      patch_size=length // tokens, hidden_size=hidden)

def unet_config(noise_type="white", prediction_target="x-pred"):
    key = (noise_type, prediction_target)
    if key not in UNET_ACTIVATIONS:
        raise ValueError("Use white/pink and x-pred/n-pred")
    return get_config("1D-U-Net", act_type=UNET_ACTIVATIONS[key])

def reconstruct(noisy, prediction, prediction_target="x-pred"):
    """Convert a raw target prediction into a clean-signal estimate."""
    if noisy.shape != prediction.shape:
        raise ValueError("noisy and prediction must have identical shapes")
    if prediction_target == "x-pred":
        return prediction
    if prediction_target == "n-pred":
        return noisy - prediction
    raise ValueError("prediction_target must be x-pred or n-pred")

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
