"""JIT-compiled CUDA kernels.

Marlin is the first kernel family. Its stable APIs are available from both
``xkernels`` and ``xkernels.marlin``.
"""

from . import marlin
from .marlin import *

__all__ = ["marlin", *marlin.__all__]
