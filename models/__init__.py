from .baseline import SRResidualUNet
from .compact_restormer import CompactRestormer


def build_model(cfg: dict):
    model_name = cfg["model"]["name"]
    in_ch = cfg["model"]["in_channels"]
    out_ch = cfg["model"]["out_channels"]
    scale = cfg["model"]["scale"]

    if model_name == "baseline":
        base_ch = cfg["model"]["baseline"]["base_ch"]
        return SRResidualUNet(in_channels=in_ch, out_channels=out_ch, base_ch=base_ch, scale=scale)
    elif model_name == "compact_restormer":
        rcfg = cfg["model"]["compact_restormer"]
        return CompactRestormer(
            in_channels=in_ch,
            out_channels=out_ch,
            scale=scale,
            embed_dim=rcfg["embed_dim"],
            num_blocks=rcfg["num_blocks"],
            heads=rcfg["heads"],
            ffn_expansion=rcfg["ffn_expansion_factor"],
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
