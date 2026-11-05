"""MetaX Catalog, primitive Measurement and layout Cost Model."""
import sys
import argparse
from dataclasses import asdict
from pathlib import Path
import json


def main():
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    catalog = commands.add_parser('catalog')
    q = catalog.add_mutually_exclusive_group()
    q.add_argument('--source')
    q.add_argument('--llvm')
    catalog.add_argument('--text')
    commands.add_parser('measurements')
    for name in ('reproduce', 'layouts'):
        c = commands.add_parser(name, help='within-shape TTGIR layout ranking')
        c.add_argument('--output', type=Path, required=True)
        c.add_argument('--device', type=int, default=0)
        c.add_argument('--sdk')
        c.add_argument('--mxcc')
        if name == 'layouts': c.add_argument('--costs', type=Path, required=True)
    for name in ('qualify', 'measure'):
        c = commands.add_parser(name)
        c.add_argument('--measurement', action='append', required=True)
        c.add_argument('--output', type=Path, required=True)
        c.add_argument('--sdk')
        c.add_argument('--mxcc')
        if name == 'measure':
            c.add_argument('--device', type=int, default=0)
            c.add_argument('--seed', type=int, default=0)
    c = commands.add_parser('analyze')
    c.add_argument('--run', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command in ('reproduce', 'layouts'):
        from reproduce import run
        print(
            json.dumps(
                run(args.output, args.device, args.sdk, args.mxcc, args.costs if args.command == 'layouts' else None),
                indent=2))
        return
    if args.command == 'catalog':
        from catalog.query import load_catalog, query_catalog, table
        print(table(query_catalog(load_catalog(), args.source, args.llvm, args.text)))
        return
    from measurement.benchmarks import list_measurements
    if args.command == 'measurements':
        print(json.dumps([s.to_dict() for s in list_measurements()], indent=2))
        return
    if args.command == 'analyze':
        from measurement.analyze import analyze
        result = analyze(args.run, args.output)
        print(json.dumps(result, indent=2))
        return
    from measurement.measure import measure, qualify_measurement
    from measurement.build import toolchain
    if len(set(args.measurement)) != len(args.measurement): p.error('duplicate measurement')
    tools = toolchain(args.sdk, args.mxcc)
    failed = False
    for name in args.measurement:
        out = args.output if len(args.measurement) == 1 else args.output / name
        if args.command == 'qualify':
            try:
                r = qualify_measurement(name, out, tools)
                print(name, json.dumps(asdict(r)))
                failed |= not r.passed
            except Exception as error:
                print(name, str(error), file=sys.stderr)
                failed = True
        else:
            try:
                r = measure(name, out, args.device, args.seed, tools)
            except Exception as error:
                print(name, str(error), file=sys.stderr)
                failed = True
                continue
            if isinstance(r, dict):
                print(json.dumps(r))
                failed = True
            else:
                print(name, r.value, r.unit)
    if failed: raise SystemExit(1)


if __name__ == '__main__': main()
