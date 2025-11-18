
"""
Helpers for re-exporting controller primitives.

The module `2smc.py` keeps the paper's naming, but it cannot be imported via
standard syntax (`from controllers.2smc import ...`) because identifiers may
not start with digits. We load it dynamically here and expose the public
symbols so downstream modules can use regular imports.
"""
from importlib import import_module


_twosmc = import_module(".2smc", __name__)

TwoSMC = _twosmc.TwoSMC
TwoSMCParams = _twosmc.TwoSMCParams
pmdc_speed_AB = _twosmc.pmdc_speed_AB
pmdc_current_AB = _twosmc.pmdc_current_AB

__all__ = [
    "TwoSMC",
    "TwoSMCParams",
    "pmdc_speed_AB",
    "pmdc_current_AB",
]
