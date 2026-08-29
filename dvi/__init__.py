"""Shared P0.5 modules for the DVI candidate-retrieval prototype: date/location
normalization, image-quality assessment, and metadata/fusion scoring. Kept as a
plain top-level package (not src/dvi) per docs/P0_5_IMPLEMENTATION.md -- this repo
has no build/packaging step, scripts and main.py import it directly off the repo
root which they already add to sys.path.
"""
