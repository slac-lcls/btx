#!/bin/bash

FUNCTION_PATH="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor.py"
# COMPS=(1 5 10 25 50 75 100 125 150 165 175 185 195 205 250)
# RANKS=(1 2 4 6 8 10 12 16)

COMPS=(100)
RANKS=(1 8)

for r in ${RANKS[@]}
do
    OUTPUT_DIR="/cds/home/h/hepworth/data/mrbm_${r}"
    mkdir -p ${OUTPUT_DIR}

    for q in ${COMPS[@]}
    do
        sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name benchmark_${r}_${q}
#SBATCH --ntasks=${r}

source /reg/g/psdm/etc/psconda.sh -py3  
export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

mpirun -n ${r} python ${FUNCTION_PATH} -e xpptut15 -r 580 -d jungfrau4M -c ${q} -b --output_dir ${OUTPUT_DIR}
echo "Benchmark with q = ${q}, r = ${r} complete."
EOF
    done
done

echo "Jobs sent to queue"






