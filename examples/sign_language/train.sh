#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=mt
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name ${EXPERIMENT}
#SBATCH --output=./logs/%j.%x.log

source ~/.bashrc
conda activate sign-language-new

export HYDRA_FULL_ERROR=1
task train

EOT
