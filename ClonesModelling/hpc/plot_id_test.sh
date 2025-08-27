#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=50:0:0
#$ -l s_rt=48:0:0
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -o plot.log

ARGS=$@
echo "$(date +'%d/%m/%Y  %H:%M:%S') -- ARGS: $ARGS"
cd /cluster/project2/clones_modelling/ClonesModelling
python3 -m ClonesModelling.id_test.plot $ARGS
