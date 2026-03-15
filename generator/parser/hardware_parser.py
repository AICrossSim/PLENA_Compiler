import argparse
import json
import re


# TODO: use the lib in tools.utils
def load_svh_settings(file_path):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """
    param_pattern = re.compile(r"\s*parameter\s+(\w+)\s*=\s*([^;]+);")
    hardware_settings = {}

    with open(file_path) as f:
        for line in f:
            match = param_pattern.match(line)
            if match:
                name, value_str = match.groups()
                value_str = value_str.strip()
                try:
                    value = int(value_str)
                except ValueError:
                    continue
                hardware_settings[name] = value
    return hardware_settings


def hardware_parser(config_file, precision_file):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """

    hardware_settings = load_svh_settings(config_file)
    precision_settings = load_svh_settings(precision_file)
    hardware_settings["wt_block_width"] = (
        precision_settings.get("WT_MX_MANT_WIDTH", 3) + precision_settings.get("WT_MX_EXP_WIDTH", 4) + 1
    ) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["kv_block_width"] = (
        precision_settings.get("KV_MX_MANT_WIDTH", 3) + precision_settings.get("KV_MX_EXP_WIDTH", 4) + 1
    ) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["act_block_width"] = (
        precision_settings.get("ACT_MX_MANT_WIDTH", 3) + precision_settings.get("ACT_MX_EXP_WIDTH", 4) + 1
    ) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["scale_width"] = (
        precision_settings.get("MX_SCALE_WIDTH", 3) + precision_settings.get("SCALE_MX_EXP_WIDTH", 4) + 1
    )
    hardware_settings["block_dim"] = hardware_settings.get("BLOCK_DIM", 4)

    return hardware_settings


def hardware_parser_to_json(config_file, precision_file, output_file=None, indent=2):
    """
    Export `hardware_parser` output to a JSON string or file.
    """
    hardware_settings = hardware_parser(config_file, precision_file)
    json_text = json.dumps(hardware_settings, indent=indent, sort_keys=True)

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(json_text)
            f.write("\n")

    return json_text


def main():
    parser = argparse.ArgumentParser(description="Export hardware parser output to JSON.")
    parser.add_argument("config_file", help="Path to the hardware configuration .svh/.sv file")
    parser.add_argument("precision_file", help="Path to the precision configuration .svh/.sv file")
    parser.add_argument(
        "-o",
        "--output",
        default="hardware_settings.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for JSON output",
    )
    args = parser.parse_args()

    hardware_parser_to_json(
        config_file=args.config_file,
        precision_file=args.precision_file,
        output_file=args.output,
        indent=args.indent,
    )
    print(f"Hardware settings JSON written to: {args.output}")


if __name__ == "__main__":
    main()
