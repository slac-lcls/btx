#!/bin/bash

FUNCTION_PATH="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor_t.py"
OUTPUT_DIR="/cds/home/h/hepworth/data/srbmt/"
COMPS=(1 5 10 20 50 75 100 125 150 165 175 180 185 190 195 200 205 250)

for q in ${COMPS[@]}
do
    sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name benchmark_${q}
#SBATCH -n 1

source /reg/g/psdm/etc/psconda.sh -py3  

export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

python ${FUNCTION_PATH} -e xpptut15 -r 580 -d jungfrau4M --components ${q} --block_size 20 --num_images 200 --output_dir ${OUTPUT_DIR}
echo "Benchmark with q = ${q} complete."
EOF
done

echo "Jobs sent to queue"






