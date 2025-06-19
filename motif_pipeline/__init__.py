"""
Top-level namespace 
"""
from importlib.metadata import version as _ver

try:          
    __version__ = _ver("msianalyzer")
except Exception:            
    __version__ = "0+local"

# ----- high-level API ------------------------------------------------
from .pipeline.driver import run_marker_pipeline, run_batch_pipeline
from .core.analysis import analyse_read
from .stats.hierarchical import run_hierarchical_tests

__all__ = [
    "run_marker_pipeline",
    "run_batch_pipeline",
    "analyse_read",
    "run_hierarchical_tests",
]
