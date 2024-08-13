import os.path

# Get the input and output destinations
import argparse
parser = argparse.ArgumentParser("Dump shader spirv")
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--var_name", required=True)
args = parser.parse_args()

if os.path.exists(args.input) != True:
    print("Failed to find input file " + args.input)
    exit(-1)

out_str = "const uint8_t {}_data []  = {{".format(args.var_name)
byte_count = 0

with open(args.input, "rb") as f:
    binary_data = f.read()
    byte_count = 0
    for val in binary_data:
        if byte_count % 10 == 0:
            out_str += "\n\t"
        byte_count += 1
        out_str += "0x{:02x}".format(val) 
        out_str += ", "

out_str += "\n};\n"

out_str = "#include <stdint.h>\n\nconst uint32_t {}_size = 0x{:x};\n".format(args.var_name, byte_count) + out_str

with open(args.output, "w") as f:
    f.write(out_str)