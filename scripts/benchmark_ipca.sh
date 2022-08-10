#!/bin/bash

FUNCTION_PATH="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor.py"
COMPS=(1 5 10 25 50 75 100 125 150 165 180 195 210 230 250)
RANKS=(4 6 8 12 16 32 64 96 128)

for r in ${RANKS[@]}
do
    OUTPUT_DIR="/cds/home/h/hepworth/data/np_2/mrbm_${r}_np"
    mkdir -p ${OUTPUT_DIR}

    for q in ${COMPS[@]}
    do
        sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name bm_${r}_${q}
#SBATCH --ntasks=${r}

source /reg/g/psdm/etc/psconda.sh -py3  
export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

mpirun -n ${r} python ${FUNCTION_PATH} -e xpptut15 -r 580 -d jungfrau4M -c ${q} -b --output_dir ${OUTPUT_DIR}
echo "Benchmark with q = ${q}, r = ${r} complete."
EOF
    done
done

echo "Jobs sent to queue"






