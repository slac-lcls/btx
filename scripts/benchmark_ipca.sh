#!/bin/bash

FUNCTION_DIR="/cds/home/h/hepworth/btx-callum/btx/processing/feature_extractor.py"
COMPS=(1 5 10 20 50 100 150 200)

for q in ${COMPS[@]}
do
    sbatch << EOF
#!/bin/bash

#SBATCH -p psanaq
#SBATCH -t 1:00:00
#SBATCH --exclusive
#SBATCH --job-name benchmark_${q}
#SBATCH -n 1

source /reg/g/psdm/etc/psconda.sh -py3  

export PYTHONPATH="${PYTHONPATH}:/cds/home/h/hepworth/btx-callum" 

mpirun -n 1 python ${FUNCTION_DIR} -e xpptut15 -r 580 -d jungfrau4M --components ${q} --block_size 20 --num_images 200
echo "Benchmark with q = ${q} complete."
EOF
done

echo "Jobs sent to queue"






