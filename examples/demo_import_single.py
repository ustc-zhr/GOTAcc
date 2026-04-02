"""
Minimal import demo for single-objective optimizers.
This file is intentionally lightweight for open-source smoke testing.
"""

from algorithms.single_objective.BOOptimizer import BOOptimizer
from algorithms.single_objective.TuRBOOptimizer import TuRBOOptimizer
from algorithms.single_objective.RCDSOptimizer import RCDSOptimizer


def main():
    print("Single-objective imports are OK:")
    print(" - BOOptimizer")
    print(" - TuRBOOptimizer")
    print(" - RCDSOptimizer")


if __name__ == "__main__":
    main()