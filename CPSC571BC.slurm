#! /bin/bash
# ====================================
#SBATCH --job-name=BCDaftHER2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=320GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
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

# python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/TumorSize_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_${SLURM_JOBID}.txt
# python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/ER_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_${SLURM_JOBID}.txt
python DAFTTrain.py "./ProcessedDataset/MRIs/" "./RawDataset/HER2_target.csv" > /home/ramin.kahidi/CPSC571/DAFT_CPSC_571/a100Output/slurmOutputs/output_${SLURM_JOBID}.txt


end=`date +%Y-%m-%d\ %H:%M:%S`

echo ==============================================
echo Start: $start
echo End: $end
