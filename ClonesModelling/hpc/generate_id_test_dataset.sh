#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=44:0:0
#$ -l s_rt=43:30:0
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -o hpc_logs/rerun_spr_25/$TASK_ID.out
SIMULATIONS_PER_RUN=25

# #$ -o hpc_logs/first_pass/$TASK_ID.out
# manually change the two previous lines to be different batch sizes for reruns to
# fix errors

wd=$(pwd)
for SIMULATION_INDEX in $(seq $((SIMULATIONS_PER_RUN * ($SGE_TASK_ID - 1) + 1)) $((SIMULATIONS_PER_RUN * $SGE_TASK_ID))); do
    JULIA_CMD_IDX=$((3 * $SIMULATION_INDEX - 2))
    OUTPUT_FILEPATH_IDX=$((3 * $SIMULATION_INDEX - 1))

    JULIA_CMD=$(head -${JULIA_CMD_IDX} simulation_commands.txt | tail -1)
    OUTPUT_FILEPATH=$(head -${OUTPUT_FILEPATH_IDX} simulation_commands.txt | tail -1)

    ERROR_FILEPATH=$PWD/hpc_logs/$SIMULATION_INDEX.out

    # if output filepath already exists and isn't empty, we've already done this one; skip
    if [ -s $OUTPUT_FILEPATH ]; then
        # check that not every line in the output file starts with "Resizing" -- if so we should delete this file and rerun
        if [ $(grep -c "^Resizing" $OUTPUT_FILEPATH) -eq $(wc -l $OUTPUT_FILEPATH | awk '{print $1}') ]; then
            echo "All lines in $OUTPUT_FILEPATH start with 'Resizing', deleting and rerunning"
            rm $OUTPUT_FILEPATH  
        else
            echo -n "$SIMULATION_INDEX "
            continue
        fi
    fi

    touch $OUTPUT_FILEPATH
    cd /cluster/project2/clones_modelling/ClonesModelling

    echo "Running $SIMULATION_INDEX"
    echo "JULIA_CMD: $JULIA_CMD"
    echo "OUTPUT_FILEPATH: $OUTPUT_FILEPATH"
    echo "ERROR_FILEPATH: $ERROR_FILEPATH"

    bash -c "$JULIA_CMD" 1>$OUTPUT_FILEPATH 2>>$ERROR_FILEPATH
    exit_code=$?
    
    if [ $exit_code -eq 1 ]; then
        if grep -q "ERROR: Unable to find compatible target in system image." $ERROR_FILEPATH; then
            JULIA_CMD=$(echo $JULIA_CMD | sed 's/--sysimage=[^ ]*//')
            
            echo "Rerunning $SIMULATION_INDEX without --sysimage flag"
            bash -c "$JULIA_CMD" 1>$OUTPUT_FILEPATH 2>>$ERROR_FILEPATH
        fi
    fi

    cd $wd
done
