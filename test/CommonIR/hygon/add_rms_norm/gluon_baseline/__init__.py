"""add_rms_norm: explicit-layout Gluon kernels; see catalog.json."""
from loader import load_kernel, variants
INDEX = 7
NAME = 'add_rms_norm'
def available():
    return variants(INDEX)
def load(name=None, reference_hash=None):
    return load_kernel(INDEX,name,reference_hash)
