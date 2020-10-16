#!/bin/bash

#SBATCH --job-name WP2_run_all_basin_scale
#SBATCH -p short-serial 
#SBATCH -o processing_output/WP2_run_basin_weighted_plotting.out 
#SBATCH -e processing_output/WP2_run_basin_weighted_plotting.err 
#SBATCH --time 10:00:00 
#SBATCH --mem 32000
cd /home/users/mmuetz/projects/cosmic/ctrl/WP2_analysis/basin_scale
pwd
bash remake_run_basin_weighted_plotting.sh
