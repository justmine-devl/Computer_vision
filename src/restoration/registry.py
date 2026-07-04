from src.restoration.dcp import DCPDehazeFilter
from src.restoration.wmgf_derain import WMGFDerainFilter
from src.restoration.morph_guided_desnow import MorphGuidedDesnowFilter, MorphGuidedDesnowConfig
from src.restoration.rbcp_desand import RBCPDesandFilter, RBCPDesandConfig
from src.restoration.bm3d_denoise import BM3DDenoiseFilter, BM3DDenoiseConfig
import yaml

def create_filter(filter_name, config_path=None):
    if filter_name.lower() == "dcp":
        from src.restoration.dcp import DCPConfig
        config = DCPConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return DCPDehazeFilter(config)
    elif filter_name.lower() == "wmgf":
        from src.restoration.wmgf_derain import WMGFConfig
        config = WMGFConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return WMGFDerainFilter(config)
    elif filter_name.lower() == "desnow":
        config = MorphGuidedDesnowConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return MorphGuidedDesnowFilter(config)
    elif filter_name.lower() in ["desand", "rbcp"]:
        config = RBCPDesandConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return RBCPDesandFilter(config)
    elif filter_name.lower() == "bm3d":
        config = BM3DDenoiseConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return BM3DDenoiseFilter(config)
    elif filter_name.lower() == "lime":
        from src.restoration.lime_delowlight import LIMEDeLowlightFilter, LIMEDeLowlightConfig
        config = LIMEDeLowlightConfig()
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    for k, v in config_data.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
        return LIMEDeLowlightFilter(config)
    elif filter_name.lower() in ["motiondeblur", "motion_deblur"]:
        from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
        config = {}
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    config = config_data
        return RichardsonLucyMotionDeblurFilter(config)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")
