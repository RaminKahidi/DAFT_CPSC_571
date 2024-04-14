#! /bin/bash
# ====================================
#SBATCH --job-name=BCDaftPreprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=124GB
#SBATCH --time=0-24:00:00
#SBATCH --partition=cpu2023
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ramin.kahidi@ucalgary.ca
# ====================================

source ~/software/init-conda
conda activate pytorch

cd "/home/ramin.kahidi/CPSC571/DAFT_CPSC_571/"

echo ==============================================
echo Hostname: `hostname`
echo Job ID: $SLURM_JOBID

echo Node: $SLURM_NODELIST

start=`date +%Y-%m-%d\ %H:%M:%S`

python MRI_Data_preprocessing.py


end=`date +%Y-%m-%d\ %H:%M:%S`

echo ==============================================
echo Start: $start
echo End: $end
