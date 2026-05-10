#!/bin/bash
cd /home/Domain-generalization-fault-diagnosis-benchmark-main

# Only run IEDGNet (others already completed)
model="IEDGNet.py"
logfile="logs/${model%.py}.log"

# Backup old log and start fresh
mv "$logfile" "${logfile}.bak" 2>/dev/null

echo "========================================" | tee -a "$logfile"
echo "Starting $model at $(date) [batch=1024, workers=16, pin_memory=True]" | tee -a "$logfile"
echo "========================================" | tee -a "$logfile"
python "$model" >> "$logfile" 2>&1
echo "Finished $model at $(date)" | tee -a "$logfile"
echo "" | tee -a "$logfile"

echo "All models finished at $(date)" | tee -a logs/all_done.log
