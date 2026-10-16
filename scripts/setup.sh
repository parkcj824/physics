#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/matplotlib}"
RESEARCH_VENV="${PHYSICS_VENV:-/workspace/.physics-venv}"
cd "$PROJECT_ROOT"
command -v uv >/dev/null
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required by this validated environment"'
if [[ ! -x "$RESEARCH_VENV/bin/python" ]]; then
    uv venv --python python3 "$RESEARCH_VENV"
fi
"$RESEARCH_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
uv pip install --python "$RESEARCH_VENV/bin/python" --require-hashes \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    -r requirements-cpu.lock
uv pip check --python "$RESEARCH_VENV/bin/python"
"$RESEARCH_VENV/bin/python" -c 'import numpy, scipy, matplotlib, pytest, torch; print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available()); assert torch.version.cuda is None'
