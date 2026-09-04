#!/bin/bash
#SBATCH --job-name=cc-pertprot-sharedallprot
#SBATCH --partition=sbnb_cpu_sphr
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00
#SBATCH --output=cc_run_%j.log
#SBATCH --error=cc_run_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=sebastian.ayala@irbbarcelona.org

# 1. Export environment variables
export HOME="/aloy/home/sayala"
export SINGULARITY_BIND="/aloy/home/sayala:/aloy/home/sayala,/aloy/scratch/sayala:/aloy/scratch/sayala,/scratch/sbnb/sayala:/scratch/sbnb/sayala"

# 2. Directory containing pertprot_cli.py, cc_pipeline.py and utils.py together
#    -- CONFIRM/EDIT this path, pertprot_cli.py needs the other two alongside it.
SCRIPT_DIR="/scratch/sbnb/sayala/protocols/scripts"

# 3. Proof-of-concept: restrict the run to a single dataset key.
#    Valid keys: dcmoa, dcmoa_deps, deps, shared_deps, allprot, shared_allprot
DATASET_KEY="shared_allprot"

# 4. Run the pipeline inside the container directly with `singularity exec`.
#    The run_chemicalchecker.sh wrapper doesn't have a mode that runs an
#    arbitrary command -- without -s it launches Jupyter Lab regardless of
#    what's passed, and with -s it opens an interactive shell. Neither runs
#    our script. So we replicate the exact `singularity exec` invocation
#    the wrapper itself used for its Jupyter mode (same env vars, same
#    --nv --cleanenv, same bind mount -- all visible in that run's .log),
#    just swapping in our own command instead of `jupyter lab`.
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH="/scratch/sbnb/sayala/chemical_checker/package"
export SINGULARITYENV_CC_CONFIG="/scratch/sbnb/sayala/chemical_checker/setup/cc_config.json"

singularity exec --nv --cleanenv \
  -B /aloy/home/sayala/chemical_checker/run_user_sing:/run/user \
  /aloy/home/sayala/chemical_checker/cc.simg \
  python "${SCRIPT_DIR}/pertprot_cli.py" --datasets "${DATASET_KEY}" --log-dir "${SCRIPT_DIR}/logs"