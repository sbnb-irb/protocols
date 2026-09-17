#!/usr/bin/env bash
# Reworked counterpart of run_compare.sh, which is kept unchanged for
# reference and still drives the original compare_cc_spaces.py.
# The singularity invocation below is byte-identical to run_compare.sh's.
# Runs compare_cc_spaces_claudecode.py inside the CC Singularity image, replicating the
# exact `singularity exec` invocation used in pertprot_batch.sh (same env
# vars, same --nv --cleanenv, same bind mounts) so it sees the identical
# filesystem/Python layout as the sbatch-submitted pipeline jobs.
#
# For scheduled/production runs, submit compare_batch.sh via `sbatch`
# instead -- this wrapper is for quick interactive checks (e.g. inside an
# `srun --pty ...` session) where going through the queue isn't necessary.
#
# Usage: mirrors compare_cc_spaces_claudecode.py's own CLI -- pass its arguments through.
#
#   ./run_compare_claudecode.sh \
#       --local-cc-dir /scratch/sbnb/sayala/protocols/local_CC_D6 \
#       --dataset-a D6.002 --label-a "DeepCoverMoA only" \
#       --dataset-b D6.007 --label-b "All Shared Proteins" \
#       --output-dir cc_comparison/D6.002_vs_D6.007
#
# Environment overrides (export before calling if your paths differ from
# pertprot_batch.sh's defaults, e.g. testing on a different cluster):
#   CC_SIMG, RUN_USER_BIND, HOME_OVERRIDE, SINGULARITY_BIND,
#   SINGULARITYENV_PYTHONPATH, SINGULARITYENV_CC_CONFIG

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Same three exports as pertprot_batch.sh step 1 -- deliberately hardcoded
# (not "${VAR:-default}") because if SINGULARITY_BIND is already set in the
# interactive shell (e.g. from a module-load/login profile covering the
# standard home/aloy paths but NOT /scratch/sbnb/sayala), the ":-" fallback
# would never fire and this path would silently stay unbound inside the
# container -- which looks exactly like "directory not found" even though
# it plainly exists from a plain host shell.
export HOME="/aloy/home/sayala"
export SINGULARITY_BIND="/aloy/home/sayala:/aloy/home/sayala,/aloy/scratch/sayala:/aloy/scratch/sayala,/scratch/sbnb/sayala:/scratch/sbnb/sayala"

# Same three SINGULARITYENV_* exports as pertprot_batch.sh step 4.
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH="${SINGULARITYENV_PYTHONPATH:-/scratch/sbnb/sayala/chemical_checker/package}"
export SINGULARITYENV_CC_CONFIG="${SINGULARITYENV_CC_CONFIG:-/scratch/sbnb/sayala/chemical_checker/setup/cc_config.json}"

CC_SIMG="${CC_SIMG:-/aloy/home/sayala/chemical_checker/cc.simg}"
RUN_USER_BIND="${RUN_USER_BIND:-/aloy/home/sayala/chemical_checker/run_user_sing:/run/user}"

if [[ ! -f "${CC_SIMG}" ]]; then
    echo "ERROR: Singularity image not found at ${CC_SIMG}" >&2
    echo "Override with CC_SIMG=/path/to/cc.simg ./run_compare_claudecode.sh ..." >&2
    exit 1
fi

# --nv --cleanenv and the /run/user bind match pertprot_batch.sh exactly.
# --cleanenv matters: without it, the host shell's PATH/PYTHONPATH (e.g.
# from the unrelated `protocols` uv project) can leak into the container
# and shadow the image's own Python/numpy.
exec singularity exec --nv --cleanenv \
    -B "${RUN_USER_BIND}" \
    "${CC_SIMG}" \
    python "${SCRIPT_DIR}/compare_cc_spaces_claudecode.py" "$@"