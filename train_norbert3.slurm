#!/bin/bash


#SBATCH --job-name=in5550
#SBATCH --account=ec30
#SBATCH --mail-type=FAIL
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=16
#SBATCH --partition=ifi_accel
#SBATCH --gpus=1


module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-transformers/4.20.1-foss-2021a-Python-3.9.5


#python3 train_norbert3.py --model_name=ltg/norbert3-xs
#python3 train_norbert3.py --model_name=ltg/norbert3-small
#python3 train_norbert3.py --model_name=ltg/norbert3-base
python3 train_norbert3.py --model_name=ltg/norbert3-large
