#!/usr/bin/env python3
"""Processes the arrays."""

from pepMeld.parse_arguments import get_process_arrays_args
from pepMeld.run_transforms import run_transforms

if __name__ == '__main__':
	args = get_process_arrays_args()
	run_transforms(args)