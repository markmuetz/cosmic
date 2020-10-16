#!/bin/bash

#SBATCH --job-name bm_figs
#SBATCH -p short-serial 
#SBATCH -o processing_output/WP2_run_basmati_figs.out 
#SBATCH -e processing_output/WP2_run_basmati_figs.err 
#SBATCH --time 10:00:00 
#SBATCH --mem 32000
cd /home/users/mmuetz/projects/cosmic/ctrl/WP2_analysis/basin_scale
pwd
bash remake_run_basmati_figs.sh
