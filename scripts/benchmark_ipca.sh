#!/bin/bash

FUNCTION_PATH="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor.py"
# COMPS=(1 5 10 25 50 75 100 125 150 165 180 195 210 230 250 270 300)
# RANKS=(4 6 8 12 16 32 64 96 128 256 384)

COMPS=(1 5 10 25 40 65 80 95 110 125 140 155 170 185 200 215 230 245 260)
RANKS=(1 64 128)

for r in ${RANKS[@]}
do
    OUTPUT_DIR="/cds/home/h/hepworth/data/ffb4_mq/mrbm_${r}"
    mkdir -p ${OUTPUT_DIR}

    for q in ${COMPS[@]}
    do
        sbatch << EOF
#!/bin/bash

#SBATCH -p ffbl4q
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name bm_${r}_${q}
#SBATCH --ntasks=${r}

source /reg/g/psdm/etc/psconda.sh -py3  
export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

mpirun -n ${r} python ${FUNCTION_PATH} --exp amo06516 --run 90 --det_type pnccdFront --num_components ${q} --benchmark_mode --downsample --bin_factor 2 --output_dir ${OUTPUT_DIR}
echo "Benchmark with q = ${q}, r = ${r} complete."
EOF
    done
done

echo "Jobs sent to queue"






