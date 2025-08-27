#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=10:0:0
#$ -l s_rt=10:0:0
#$ -R y
#$ -S /bin/bash

ARGS=$@
echo ARGS: $ARGS
python3 -m ClonesModelling.id_test.simulation_commands $ARGS
