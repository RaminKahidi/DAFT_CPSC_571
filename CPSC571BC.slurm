#! /bin/bash
# ====================================
#SBATCH --job-name=BCDaft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=320GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
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

# python DataloaderTester.py
python DAFTTrain.py


end=`date +%Y-%m-%d\ %H:%M:%S`

echo ==============================================
echo Start: $start
echo End: $end
