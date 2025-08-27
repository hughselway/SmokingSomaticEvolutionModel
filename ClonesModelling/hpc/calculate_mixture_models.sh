#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -j y
#$ -R y
#$ -wd /cluster/project2/clones_modelling/ClonesModelling
#$ -o /SAN/medic/hselway_omics/identifiability_test/calculate_mixture_models.log

ARGS=$@
echo ARGS: $ARGS

python3 -m ClonesModelling.hpc.calculate_mixture_models $ARGS

echo done
