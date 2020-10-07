#!/bin/bash

#SBATCH --job-name WP2_run_all_basin_scale
#SBATCH -p short-serial 
#SBATCH -o processing_output/WP2_run_all_%J.out 
#SBATCH -e processing_output/WP2_run_all_%J.err 
#SBATCH --time 10:00:00 
#SBATCH --mem 32000

cd /home/users/mmuetz/cosmic_ctrl/WP2_analysis/basin_scale
bash remake_run_all.sh
