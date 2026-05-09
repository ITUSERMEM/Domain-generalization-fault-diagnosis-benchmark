#!/bin/bash
cd /home/Domain-generalization-fault-diagnosis-benchmark-main

MODELS=("ERM.py" "DANN.py" "DDC.py" "DCORAL.py" "CCDG.py" "CNN-C.py" "DGNIS.py" "IEDGNet.py")

for model in "${MODELS[@]}"; do
    logfile="logs/${model%.py}.log"
    echo "========================================" | tee -a "$logfile"
    echo "Starting $model at $(date)" | tee -a "$logfile"
    echo "========================================" | tee -a "$logfile"
    python "$model" >> "$logfile" 2>&1
    echo "Finished $model at $(date)" | tee -a "$logfile"
    echo "" | tee -a "$logfile"
done

echo "All models finished at $(date)" | tee -a logs/all_done.log
