#!/bin/bash


#SBATCH --job-name=in5550
#SBATCH --account=ec30
#SBATCH --mail-type=FAIL
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=6
#SBATCH --partition=ifi_accel
#SBATCH --gpus=1


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
module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5
module load nlpl-simple_elmo/0.9.1-foss-2021a-Python-3.9.5
# print information (optional)
# echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
# python3 train.py --embeddings "/fp/projects01/ec30/models/static/82/model.bin" --path "/fp/projects01/ec30/IN5550/labs/06/stanford_sentiment_binary.tsv.gz" ${@}


python3 predict_on_test.py --test '/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz'