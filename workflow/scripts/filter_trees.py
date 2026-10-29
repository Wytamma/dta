#!/usr/bin/env python3
"""Drop the leading fraction of trees from a NEXUS tree file.

The tree set an upstream analysis provides is a posterior sample, and its first
entries are the unconverged start of that chain.  BEAST's
``empiricalTreeDistributionModel`` samples uniformly from whatever file it is
given, so burn-in left in the input is not harmless: every retained draw has the
same chance of landing on a pre-convergence tree.

Everything except the tree statements (header, ``translate`` block, ``End;``) is
copied verbatim, so the output is a valid NEXUS file with the same taxon
numbering.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: a tree statement starts at the beginning of its line
TREE_START = re.compile(r"^\s*tree\s")


def tree_count(path) -> int:
    """Number of tree statements in ``path`` (a line scan, nothing is parsed)."""
    with Path(path).open() as handle:
        return sum(1 for line in handle if TREE_START.match(line))


def filter_trees(source, target, burnin_fraction: float) -> tuple[int, int]:
    """Write ``source`` to ``target`` without its first ``burnin_fraction`` trees.

    Returns ``(kept, total)``.  At least one tree is always kept.
    """
    source, target = Path(source), Path(target)
    total = tree_count(source)
    if total == 0:
        raise ValueError(f"{source}: no tree statements found")
    skip = min(int(total * max(burnin_fraction, 0.0)), total - 1)
    kept = 0
    with source.open() as reader, target.open("w") as writer:
        index = 0
        for line in reader:
            if TREE_START.match(line) is None:
                writer.write(line)
                continue
            index += 1
            if index <= skip:
                continue
            writer.write(line)
            kept += 1
    return kept, total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trees", required=True, help="input NEXUS tree file")
    parser.add_argument("--output", required=True, help="NEXUS file to write")
    parser.add_argument(
        "--burnin-fraction",
        type=float,
        default=0.1,
        help="fraction of the leading trees to drop (default: 0.1)",
    )
    args = parser.parse_args(argv)
    kept, total = filter_trees(args.trees, args.output, args.burnin_fraction)
    print(f"{args.output}: kept {kept}/{total} trees (dropped {total - kept})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
