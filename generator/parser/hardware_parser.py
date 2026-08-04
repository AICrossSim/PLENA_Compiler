import re


def _matrix_element_width(precision_settings, prefix):
    if precision_settings.get(f"{prefix}_MX_INT_ENABLE", 0):
        key = f"{prefix}_MX_INT_WIDTH"
        width = precision_settings.get(key)
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"{key} must be a positive integer when MXINT is enabled")
        return width

    if prefix == "ACT":
        mantissa = precision_settings.get(
            "ACT_MXFP_MANT_WIDTH",
            precision_settings.get("ACT_MX_MANT_WIDTH", 3),
        )
        exponent = precision_settings.get(
            "ACT_MXFP_EXP_WIDTH",
            precision_settings.get("ACT_MX_EXP_WIDTH", 4),
        )
    else:
        mantissa = precision_settings.get(f"{prefix}_MX_MANT_WIDTH", 3)
        exponent = precision_settings.get(f"{prefix}_MX_EXP_WIDTH", 4)
    return mantissa + exponent + 1


# TODO: use the lib in tools.utils
def load_svh_settings(file_path):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """
    param_pattern = re.compile(r"\s*(?:localparam|parameter)\s+(\w+)\s*=\s*([^;]+);")
    hardware_settings = {}

    with open(file_path) as f:
        for line in f:
            match = param_pattern.match(line)
            if match:
                name, value_str = match.groups()
                value_str = value_str.strip()
                # Try integer conversion first
                try:
                    value = int(value_str)
                except ValueError:
                    # Fallback to raw string (could be expression or real number)
                    continue
                hardware_settings[name] = value
    return hardware_settings


def hardware_parser(config_file, precision_file):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """

    hardware_settings = load_svh_settings(config_file)
    precision_settings = load_svh_settings(precision_file)
    block_dim = precision_settings.get(
        "BLOCK_DIM",
        hardware_settings.get("BLOCK_DIM", 4),
    )
    hardware_settings["wt_block_width"] = (
        _matrix_element_width(precision_settings, "WT") * block_dim
    )
    hardware_settings["kv_block_width"] = (
        _matrix_element_width(precision_settings, "KV") * block_dim
    )
    hardware_settings["act_block_width"] = (
        _matrix_element_width(precision_settings, "ACT") * block_dim
    )
    if "SCALE_MX_EXP_WIDTH" in precision_settings:
        hardware_settings["scale_width"] = (
            precision_settings.get("MX_SCALE_WIDTH", 3)
            + precision_settings["SCALE_MX_EXP_WIDTH"]
            + 1
        )
    else:
        hardware_settings["scale_width"] = precision_settings.get(
            "MX_SCALE_WIDTH",
            8,
        )
    hardware_settings["block_dim"] = block_dim

    return hardware_settings
