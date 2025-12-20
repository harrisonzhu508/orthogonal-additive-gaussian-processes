#!/bin/bash

# run_all_linear.sh
#
# Launches all 10 tree × interaction_level combinations in parallel
# Each job runs as a separate R process with its own log file
#
# Usage:
#   bash speech_phylo/final_scripts/run_all_linear.sh

cd /home/people/habhuz/orthogonal-additive-gaussian-processes

# Create logs directory
LOG_DIR="speech_phylo/final_phyloregression_results/logs"
mkdir -p "$LOG_DIR"

# Define trees and interaction levels
TREES=("long_v3_CCD0.0.25" "heggarty2024")
# LEVELS=("main_only" "secondorder" "thirdorder" "fourthorder" "fifthorder")
LEVELS=("main_only" "secondorder")

echo "═══════════════════════════════════════════════════════════════════"
echo "Launching all combinations in parallel..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Launch all jobs
for tree in "${TREES[@]}"; do
    for level in "${LEVELS[@]}"; do
        LOG_FILE="$LOG_DIR/${tree}_${level}.log"
        echo "Starting: $tree × $level → $LOG_FILE"
        
        nohup Rscript speech_phylo/final_scripts/run_linear_single.R "$tree" "$level" \
            > "$LOG_FILE" 2>&1 &
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "All 10 jobs launched! Check status with:"
echo "  ps aux | grep run_linear_single"
echo ""
echo "View logs with:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Or check individual logs:"
for tree in "${TREES[@]}"; do
    for level in "${LEVELS[@]}"; do
        echo "  tail -f $LOG_DIR/${tree}_${level}.log"
    done
done
echo "═══════════════════════════════════════════════════════════════════"
