#!/bin/sh
#SBATCH --job-name=train
#SBATCH -t 30-00:00:00
#SBATCH --partition=compsci
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

echo get_nyuv2_list
python3 create_nyuv2_filelist.py