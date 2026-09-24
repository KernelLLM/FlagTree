"""rms_norm_w8a16_fp8: explicit-layout Gluon kernels; see catalog.json."""
from loader import load_kernel, variants
INDEX = 11
NAME = 'rms_norm_w8a16_fp8'
def available():
    return variants(INDEX)
def load(name=None, reference_hash=None):
    return load_kernel(INDEX,name,reference_hash)
