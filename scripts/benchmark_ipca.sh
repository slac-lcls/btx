#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to run naive iPCA benchmark.

  OPTIONS:
    -h|--help
      Definitions of options
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -h|--help)
      usage
      exit
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

FUNCTION_DIR="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor.py"
COMPS=(1 5 10 20 50 100 150 200)

sbatch << EOF
#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name extractor_benchmark
#SBATCH --ntasks=1

source /reg/g/psdm/etc/psconda.sh -py3  

export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

for q in ${COMPS[@]}; do;
    mpirun -n 1 python ${FUNCTION_DIR} -e xpptut15 -r 580 -d jungfrau4M --components ${q} --block_size 20 --num_images 200
    echo "Benchmark with q = ${q} complete."
EOF
echo "Job sent to queue"






