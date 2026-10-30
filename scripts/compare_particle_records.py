#!/usr/bin/env python3
"""Compare native-endian AMReX double-precision particle output exactly.

Usage: python3 scripts/compare_particle_records.py LEFT_POPULATION RIGHT_POPULATION

Supports the AMReX 2.0 plotfile and 2.1 checkpoint formats used by ADR-0002.
Compares records by stored identity, independently of grid and file ordering.
This checks particle data and metadata, not mesh fields or physical accuracy.
"""
import argparse
import json
import math
import pathlib
import struct
import sys


def read(path):
    path = pathlib.Path(path)
    lines = iter((path / 'Header').read_text().splitlines())
    version = next(lines)
    assert version in ('Version_Two_Dot_One_double', 'Version_Two_Dot_Zero_double'), version
    dim = int(next(lines))
    nr = int(next(lines))
    real_names = [next(lines) for _ in range(nr)]
    ni = int(next(lines))
    int_names = [next(lines) for _ in range(ni)]
    assert int(next(lines)) == 1
    count = int(next(lines))
    next_id = int(next(lines))
    finest = int(next(lines))
    grids = [int(next(lines)) for _ in range(finest + 1)]
    records = {}
    for lev, ng in enumerate(grids):
        for _ in range(ng):
            file, n, offset = map(int, next(lines).split())
            if n == 0:
                continue
            with (path / f'Level_{lev}' / f'DATA_{file:05d}').open('rb') as stream:
                stream.seek(offset)
                ints = struct.unpack(f'={n * (2 + ni)}i', stream.read(4 * n * (2 + ni)))
                reals = struct.unpack(f'={n * (dim + nr)}d', stream.read(8 * n * (dim + nr)))
            for i in range(n):
                ii = ints[i*(2+ni):(i+1)*(2+ni)]
                rr = reals[i*(dim+nr):(i+1)*(dim+nr)]
                identity = ii[:2]
                assert identity not in records, identity
                records[identity] = (ii[2:], rr)
    assert len(records) == count, (len(records), count)
    return (dim, real_names, int_names, next_id), records

def compare(left, right, roundoff_factor=0, *, require_equal_next_id=True):
    """Compare each real field against factor * epsilon * its global max norm.

    The zero default retains exact comparison. Counts, identities, integer data,
    schema, and units always require exact agreement. The next-ID counter is
    exact by default; a caller comparing shared versus per-type counters may
    validate their lifecycle requirements separately. Nonfinite data fail even
    when both inputs contain the same infinity or NaN.
    """
    if not math.isfinite(roundoff_factor) or roundoff_factor < 0:
        raise ValueError("Roundoff factor must be finite and nonnegative")
    left, right = map(pathlib.Path, (left, right))
    lmeta, lrecords = read(left)
    rmeta, rrecords = read(right)
    if lmeta[:3] != rmeta[:3] or (require_equal_next_id and lmeta[3] != rmeta[3]):
        raise ValueError(f"Particle metadata differ: {lmeta} != {rmeta}")
    if lrecords.keys() != rrecords.keys():
        raise ValueError("Particle IDs/CPU identities differ")
    names = [f"position_{d}" for d in range(lmeta[0])] + lmeta[1]
    scales = [0.0] * len(names)
    differences = [0.0] * len(names)
    for identity, (ints, reals) in lrecords.items():
        other_ints, other_reals = rrecords[identity]
        if ints != other_ints:
            raise ValueError(f"Integer fields differ for {identity}")
        for i, (a, b) in enumerate(zip(reals, other_reals)):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Nonfinite {names[i]} for {identity}")
            scales[i] = max(scales[i], abs(a), abs(b))
            differences[i] = max(differences[i], abs(a - b))
    fields = {}
    for name, scale, difference in zip(names, scales, differences):
        tolerance = roundoff_factor * sys.float_info.epsilon * scale
        fields[name] = {"max_absolute_difference": difference, "scale": scale,
                        "absolute_tolerance": tolerance, "passed": difference <= tolerance}
    left_units, right_units = left / 'Fields.yaml', right / 'Fields.yaml'
    if left_units.exists() != right_units.exists() or (left_units.exists() and left_units.read_bytes() != right_units.read_bytes()):
        raise ValueError("Particle units metadata differ")
    return {"records": len(lrecords), "roundoff_factor": roundoff_factor,
            "next_ids": [lmeta[3], rmeta[3]],
            "fields": fields, "passed": all(field["passed"] for field in fields.values())}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=pathlib.Path)
    parser.add_argument('right', type=pathlib.Path)
    parser.add_argument('--roundoff-factor', type=float, default=0,
                        help='Allowed epsilon multiples of each field max norm; default 0 requires exact values')
    parser.add_argument('--json', type=pathlib.Path, help='Write per-field comparison evidence')
    args = parser.parse_args(argv)
    result = compare(args.left, args.right, args.roundoff_factor)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError(f"Particle real fields exceed tolerance: {result['fields']}")
    print(f"Agreement: {result['records']} records; roundoff factor {args.roundoff_factor}; exact identities, integers, metadata, and units")


if __name__ == "__main__":
    main(sys.argv[1:])
