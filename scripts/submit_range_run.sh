#!/bin/bash

usage()
{
cat << EOF
$(basename "$0"):
  Script to launch python scripts from the eLog

  OPTIONS:
    -h|--help
      Definition of options
    -f|--facility
      Facility where we are running at
    -q|--queue
      Queue to use on SLURM
    -n|--ncores
      Number of cores
    -c|--config_file
      Input config file
    -e|--experiment_name
      Experiment Name
    -r|--run_number
      Run Number
    -s|--run_number_stop
      Last Run Number to consider
    -t|--task
      Task name
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
    -f|--facility)
      FACILITY="$2"
      shift
      shift
      ;;
    -q|--queue)
      QUEUE="$2"
      shift
      shift
      ;;
    -n|--ncores)
      CORES="$2"
      shift
      shift
      ;;
    -c|--config_file)
      CONFIGFILE="$2"
      shift
      shift
      ;;
    -e|--experiment_name)
      EXPERIMENT=$2
      shift
      shift
      ;;
    -r|--run_number)
      RUN_NUM=$2
      shift
      shift
      ;;
    -s|--run_number_stop)
      RUN_NUM_STOP=$2
      shift
      shift
      ;;
    -t|--task)
      TASK="$2"
      shift
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
    esac
done
set -- "${POSITIONAL[@]}"

FACILITY=${FACILITY:='SRCF_FFB'}
QUEUE=${QUEUE:='ffbh3q'}
CORES=${CORES:=1}
EXPERIMENT=${EXPERIMENT:='None'}
RUN_NUM=${RUN_NUM:='None'}
RUN_NUM_STOP=${RUN_NUM_STOP:=$RUN_NUM}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ${RUN_NUM} != 'None' ]; then
  while [ $RUN_NUM -le $RUN_NUM_STOP ]
  do
    echo "${SCRIPT_DIR}/elog_submit.sh -f $FACILITY -q $QUEUE -n $CORES -c $CONFIGFILE -e $EXPERIMENT -r $RUN_NUM -t $TASK"
    ${SCRIPT_DIR}/elog_submit.sh -f $FACILITY -q $QUEUE -n $CORES -c $CONFIGFILE -e $EXPERIMENT -r $RUN_NUM -t $TASK
    RUN_NUM=$(( $RUN_NUM + 1 ))
  done
else
  echo "${SCRIPT_DIR}/elog_submit.sh -f $FACILITY -q $QUEUE -n $CORES -c $CONFIGFILE -e $EXPERIMENT -r $RUN_NUM -t $TASK"
  ${SCRIPT_DIR}/elog_submit.sh -f $FACILITY -q $QUEUE -n $CORES -c $CONFIGFILE -e $EXPERIMENT -r $RUN_NUM -t $TASK
fi