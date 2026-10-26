"""Locations of the code and data folders, so that every step can be run from any working directory.
Outputs are always written to the current working directory (run_all.py uses output/)."""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('CRH_DATA_DIR', os.path.join(CODE, os.pardir, 'data'))


def code_src(name):
    """Source text of another step, which some steps reuse with exec()."""
    with open(os.path.join(CODE, name), encoding='utf-8') as f:
        return f.read()
