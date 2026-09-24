#!/usr/bin/env python
"""Removed: the AZ embedding charts are ``az_inspect.py project --chart``
(the 2D scatter of trained rows; --method pca|tsne) and ``az_inspect.py drift
--chart`` (the movement-PC heatmap)."""

import sys

if __name__ == "__main__":
    sys.exit("az_embed_viz.py was removed; use `az_inspect.py project --chart` "
             "(scatter; --method pca|tsne) or `az_inspect.py drift --chart` "
             "(movement map)")
