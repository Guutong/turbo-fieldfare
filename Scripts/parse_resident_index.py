#!/usr/bin/env python3
"""Parse the resident index from Qwen3.6 turbo model weights bin.

Resident index layout:
  Offset 0-23:  24-byte header (indexSize u64, residentSize u64, entryCount u64, all LE)
  Offset 24+:   entry table — 72 bytes per entry
    nameOffset  u32  @  0   (absolute file offset into names section)
    nameLength  u16  @  4
    dtype       u8   @  6
    pad         u8   @  7
    fileOffset  u64  @  8
    sizeBytes   u64  @ 16
    shape       4*u32@ 24
    scaleOffset u64  @ 40
    scaleSize   u64  @ 48
    biasOffset  u64  @ 56
    biasSize    u64  @ 64
  After entry table: names stored as FIXED-LENGTH blocks (nameLength bytes each, NO null terminator).
"""

import struct
import sys

ENTRY_FMT = "<I H B B Q Q 4I 4Q"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)
HEADER_FMT = "<QQQ"
assert ENTRY_SIZE == 72, f"entry size {ENTRY_SIZE} != 72"

DTYPE_MAP = {
    0: "unknown", 1: "f16", 2: "f32", 3: "f64",
    4: "u8", 5: "i8", 6: "u16", 7: "i16",
    8: "u32", 9: "i32", 10: "u64", 11: "i64",
    12: "bf16", 13: "f8", 14: "e4m3", 15: "e5m2",
}


def parse_header(data):
    return struct.unpack_from(HEADER_FMT, data, 0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen36.gturbo/model_weights.bin"

    print(f"Opening {path} ...")
    with open(path, "rb") as f:
        data = f.read()

    print(f"File size: {len(data)} bytes ({len(data)/1024**2:.1f} MB)")

    index_size, resident_size, entry_count = parse_header(data)
    print(f"\nHeader: indexSize={index_size}  residentSize={resident_size}  entryCount={entry_count}")
    print(f"Entry table ends at: 24 + {entry_count} * 72 = {24 + entry_count * 72}")

    target_name = "language_model.model.layers.0.linear_attn.conv1d.weight"
    targets = set()
    target_details = {}
    conv1d_entries = []

    for i in range(entry_count):
        off = 24 + i * 72
        fields = struct.unpack_from(ENTRY_FMT, data, off)
        name_offset = fields[0]
        name_length = fields[1]
        dtype_byte = fields[2]
        file_offset = fields[4]
        size_bytes = fields[5]
        shape = tuple(fields[6:10])
        scale_offset = fields[10]
        scale_size = fields[11]
        bias_offset = fields[12]
        bias_size = fields[13]

        dtype_str = DTYPE_MAP.get(dtype_byte, f"custom({dtype_byte})")

        # Names are fixed-length, not null-terminated
        name = data[name_offset:name_offset + name_length].decode("utf-8")

        if name == target_name:
            targets.add(name)
            target_details[target_name] = {
                "index": i, "name": name, "dtype": dtype_str,
                "fileOffset": file_offset, "sizeBytes": size_bytes,
                "shape": shape,
                "scaleOffset": scale_offset, "scaleSize": scale_size,
                "biasOffset": bias_offset, "biasSize": bias_size,
            }

        if "conv1d" in name:
            conv1d_entries.append((i, name))

    # Print target tensor detail
    if target_name in target_details:
        d = target_details[target_name]
        print("\n" + "=" * 72)
        print(f"TENSOR: {d['name']}")
        print("=" * 72)
        print(f"  Index in table:    {d['index']}")
        print(f"  dtype:             {d['dtype']}")
        print(f"  fileOffset:        {d['fileOffset']}  (0x{d['fileOffset']:x})")
        print(f"  sizeBytes:         {d['sizeBytes']}")
        print(f"  shape[0..3]:       {list(d['shape'])}")
        print(f"  scaleOffset:       {d['scaleOffset']}  (0x{d['scaleOffset']:x})")
        print(f"  scaleSize:         {d['scaleSize']}")
        print(f"  biasOffset:        {d['biasOffset']}  (0x{d['biasOffset']:x})")
        print(f"  biasSize:          {d['biasSize']}")
        print("=" * 72)

    # Print ALL conv1d tensors
    print(f"\nALL {len(conv1d_entries)} TENSORS CONTAINING 'conv1d':")
    for idx, name in conv1d_entries:
        print(f"  [{idx:>3}] {name}")

    # Layer summary
    layers = set()
    for _, name in conv1d_entries:
        parts = name.split(".")
        li = parts.index("layers")
        layers.add(int(parts[li + 1]))
    print(f"\nConv1d weight layers present: {sorted(layers)}")


if __name__ == "__main__":
    main()
