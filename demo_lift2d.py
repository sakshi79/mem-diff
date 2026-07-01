"""
demo_lift2d.py  (thin wrapper)
==============================
Delegates directly to the self-contained lift2d.py env + demo.

Usage
-----
  python demo_lift2d.py                     # interactive, no recording
  python demo_lift2d.py -o data/lift.zarr   # interactive + record
"""
from envs2d.lift2d import main

if __name__ == "__main__":
    main()
