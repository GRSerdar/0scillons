"""Cluster-agnostic filesystem layout for the oscillon project.

The code package lives at ``February26Oscillons/Feb26``.  The repository
root two levels above that contains the ``oscillon_runs_data`` symlink to
the large simulation store (on Tycho: Lustre under
``/lustre/astro/syildiz/VSC_data/oscillon_runs``).

Override with environment variables if needed:

* ``OSCILLON_RUN_DATA``  – directory of production run folders
* ``OSCILLON_GAUGE_DATA`` – directory of gauge-testing run folders
* ``OSCILLON_DATA_ROOT``  – parent of ``oscillon_runs`` / ``gauge_testing``
* ``VSC_DATA``            – legacy KU Leuven name; still honoured if set
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parents[1]


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _run_data() -> Path:
    env = os.environ.get("OSCILLON_RUN_DATA")
    if env:
        return Path(env)

    vsc = os.environ.get("VSC_DATA")
    if vsc:
        return Path(vsc) / "oscillon_runs"

    linked = REPO_ROOT / "oscillon_runs_data"
    found = _first_existing(linked, Path("/lustre/astro/syildiz/VSC_data/oscillon_runs"))
    return found if found is not None else linked


def _data_root() -> Path:
    env = os.environ.get("OSCILLON_DATA_ROOT")
    if env:
        return Path(env)

    vsc = os.environ.get("VSC_DATA")
    if vsc:
        return Path(vsc)

    try:
        return RUN_DATA.resolve().parent
    except OSError:
        return REPO_ROOT


RUN_DATA = _run_data()
DATA_ROOT = _data_root()
GAUGE_DATA = Path(
    os.environ.get("OSCILLON_GAUGE_DATA", str(DATA_ROOT / "gauge_testing"))
)
OSCILLATON_CSV_DIR = CODE_ROOT / "initialdata" / "oscillaton"


def add_code_root_to_sys_path() -> str:
    """Put the Feb26 package root on ``sys.path`` and return it as a string."""
    root = str(CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
