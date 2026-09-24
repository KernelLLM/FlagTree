"""mhc_pre: explicit-layout Gluon kernels; see catalog.json."""
from loader import load_kernel, variants
INDEX = 16
NAME = 'mhc_pre'
def available():
    return variants(INDEX)
def load(name=None, reference_hash=None):
    return load_kernel(INDEX,name,reference_hash)
