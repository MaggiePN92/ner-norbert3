#!/bin/bash
#SBATCH --job-name=SST2
#SBATCH --account=ec30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
# #SBATCH --partition=accel
# #SBATCH --gpus=1

# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
module load nlpl-pytorch/1.11.0-foss-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-gensim/4.3.1-foss-2021a-Python-3.9.5

# print information (optional)
# echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
# python3 train.py --embeddings "/fp/projects01/ec30/models/static/82/model.bin" --path "/fp/projects01/ec30/IN5550/labs/06/stanford_sentiment_binary.tsv.gz" ${@}


python3 linear_model.py --lr 2e-1  ${@}  # - pos, mean