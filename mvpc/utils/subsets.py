"""
subsets.py

Utility functions for generating conditioning sets.
"""

from itertools import combinations

def all_subsets(neighbors, size):
    """
    Generate all subsets of given size.
    """
    return combinations(neighbors, size)
