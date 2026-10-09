#!/bin/bash
# Run any command inside the Chemical Checker Singularity container, either
# directly or as a SLURM job (the #SBATCH lines below are plain comments to bash).
#
#   bash   scripts/chemcheck_exec.sh python -m chemcheck_protocols fit-signatures --config my.yaml --max-stage sign2
#   sbatch scripts/chemcheck_exec.sh python -m chemcheck_protocols fit-signatures --config my.yaml --start-stage sign3
#   bash   scripts/chemcheck_exec.sh python -m pytest
#
# Environment variables:
#   CC_IMAGE   (required) Singularity image with chemicalchecker, e.g. cc.simg
#   CC_CONFIG  (optional) chemicalchecker cc_config.json; can also be set in the run config
#   CC_REPO    (optional) chemical_checker clone whose package/ is imported instead of the image's copy
#   CC_BIND    (optional) extra comma-separated bind paths (data, CC instances, CC_REPO, CC_CONFIG...);
#              this repository is always bound
#
# SLURM resources: the defaults below can be overridden per job with sbatch flags
# (sbatch --partition=... --gres=gpu:1 --time=...) or per shell with SBATCH_PARTITION,
# SBATCH_QOS, SBATCH_TIMELIMIT, ... environment variables.
#
#SBATCH --job-name=chemcheck
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARGS...]   (see the header of this script)" >&2
    exit 2
fi
if [[ -z "${CC_IMAGE:-}" || ! -f "${CC_IMAGE}" ]]; then
    echo "error: set CC_IMAGE to the Chemical Checker Singularity image (got '${CC_IMAGE:-}')" >&2
    exit 2
fi

# sbatch runs a spooled copy of this script, so locate the original through SLURM.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -d "${script_dir}/../src/chemcheck_protocols" && -n "${SLURM_JOB_ID:-}" ]]; then
    script_path=$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
    script_dir=$(dirname "${script_path}")
fi
project_dir=$(cd "${script_dir}/.." && pwd)

# --cleanenv keeps host variables out of the container, so everything it needs
# is passed explicitly; SINGULARITY_BIND is set, not appended to, so a host
# default from a login profile cannot leave paths unbound.
export SINGULARITY_BIND="${project_dir}${CC_BIND:+,${CC_BIND}}"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH="${project_dir}/src${CC_REPO:+:${CC_REPO}/package}"
if [[ -n "${CC_CONFIG:-}" ]]; then
    export SINGULARITYENV_CC_CONFIG="${CC_CONFIG}"
fi
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    # Keep numpy/TensorFlow threads within the job's CPUs.
    export SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

gpu_flag=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    gpu_flag=(--nv)
fi

echo "[chemcheck_exec] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-none} image=${CC_IMAGE} gpu=${gpu_flag[*]:-no}" >&2
echo "[chemcheck_exec] command: $*" >&2
exec singularity exec --cleanenv "${gpu_flag[@]}" "${CC_IMAGE}" "$@"
