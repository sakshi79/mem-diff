"""
demo_pickplace2d.py  (thin wrapper)
===================================
Delegates directly to the self-contained pickplace2d.py env + demo.

Usage
-----
  python demo_pickplace2d.py                          # interactive, no recording
  python demo_pickplace2d.py -o data/pickplace.zarr   # interactive + record
"""
from envs2d.pickplace2d import main

if __name__ == "__main__":
    main()
