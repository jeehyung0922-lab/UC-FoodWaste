"""CLI entry point; a frozen GUI executable uses this engine as well."""
import argparse
import json
from pathlib import Path
import sys

from assets import build_bundle
from engine import compare, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    bundle=sub.add_parser("build-assets")
    for arg in ("split","masks","polygons","out"):
        bundle.add_argument("--"+arg,type=Path,required=True)
    ev=sub.add_parser("evaluate")
    for arg in ("maps","assets","out"):
        ev.add_argument("--"+arg,type=Path,required=True)
    ev.add_argument("--model",required=True)
    ev.add_argument("--metadata",type=Path)
    cmp=sub.add_parser("compare")
    cmp.add_argument("--out",type=Path,required=True)
    cmp.add_argument("results",nargs="+",type=Path)
    args=parser.parse_args()
    try:
        if args.command=="build-assets":
            images,regions=build_bundle(args.split,args.masks,args.polygons,args.out)
            print(f"{images} Validation images; {regions} source polygons")
        elif args.command=="evaluate":
            print(json.dumps(evaluate(args.model,args.maps,args.assets,args.out,args.metadata),indent=2))
        else:
            compare(args.results,args.out)
    except (OSError,ValueError,TypeError,KeyError,SyntaxError) as exc:
        parser.exit(2,f"Evaluation stopped: {exc}\n")


if __name__=="__main__":
    main()
