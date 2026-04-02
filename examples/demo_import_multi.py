"""
Minimal import demo for multi-objective optimizers.
"""

from algorithms.multi_objective.MOBOOptimizer import MOBOOptimizer
from algorithms.multi_objective.MGGPO import MGGPO
from algorithms.multi_objective.MOPSOOptimizer import MOPSOOptimizer
from algorithms.multi_objective.NSGA2Optimizer import NSGA2Optimizer


def main():
    print("Multi-objective imports are OK:")
    print(" - MOBOOptimizer")
    print(" - MGGPO")
    print(" - MOPSOOptimizer")
    print(" - NSGA2Optimizer")


if __name__ == "__main__":
    main()