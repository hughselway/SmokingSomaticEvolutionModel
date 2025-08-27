#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=41:0:0
#$ -l s_rt=40:0:0
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -o group_$TASK_ID.log

ARGS=$@

if [ -f failed.txt ]; then
    # it's comma-separated list of failed group numbers, take the SGE_TASK_ID-th one
    GROUP_NUMBER=$(cat failed.txt | cut -d, -f$SGE_TASK_ID)
    # otherwise GROUP_NUMBER is the same as SGE_TASK_ID
else
    GROUP_NUMBER=$SGE_TASK_ID
fi

cd /cluster/project2/clones_modelling/ClonesModelling
python3 -m ClonesModelling.id_test.calculate_distances $ARGS --group_number $GROUP_NUMBER
