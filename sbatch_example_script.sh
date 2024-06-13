#!/bin/bash
#SBATCH --chdir /cluster/courses/digital_humans/datasets/team_7/code/3dgs-avatar-release
#SBATCH --job-name 3dgs-smpl
echo "job name assigned"
#SBATCH --account digital_humans
#SBATCH --partition gpu
echo "account assigned"
#SBATCH --time=48:00:00
#SBATCH -o /cluster/courses/digital_humans/datasets/team_7/code/3dgs-avatar-release/job_log/slurm_output_%j.out
echo "output assigned"
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
# source <conda executable> and activate environment
source /cluster/courses/digital_humans/datasets/team_7/tianhao/miniconda3/etc/profile.d/conda.sh
conda activate 3dgs-avatar
# go to root directory of your code
cd /cluster/courses/digital_humans/datasets/team_7/code/3dgs-avatar-release/

# Run your experiment
# SMPL
# python train.py dataset=ablation_study/smpl/x_humans_00036_take1 option=iter30k pose_correction=none
# python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ablation_study/smpl/x_humans_00036_take1
# python train.py dataset=ablation_study/smpl/x_humans_00036_take6 option=iter30k pose_correction=none
# python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ablation_study/smpl/x_humans_00036_take6
# python train.py dataset=ablation_study/smpl/x_humans_00036_take9 option=iter30k pose_correction=none
# python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ablation_study/smpl/x_humans_00036_take9
# python train.py dataset=ablation_study/smpl/x_humans_00036_take169 option=iter30k pose_correction=none
# python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ablation_study/smpl/x_humans_00036_take9

# SMPL-X
# Individual takes
python train.py dataset=ablation_study/smplx/x_humans_00036_take1 non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none option=iter30k 
python render.py mode=test dataset.test_mode=pose non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none dataset=ablation_study/smplx/x_humans_00036_take1

python train.py dataset=ablation_study/smplx/x_humans_00036_take6 non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none option=iter30k
python render.py mode=test dataset.test_mode=pose non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none dataset=ablation_study/smplx/x_humans_00036_take6

python train.py dataset=ablation_study/smplx/x_humans_00036_take9 non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none option=iter30k
python render.py mode=test dataset.test_mode=pose non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none dataset=ablation_study/smplx/x_humans_00036_take9

# Takes combined
python train.py dataset=ablation_study/smplx/x_humans_00036_take169 non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none option=iter30k
python render.py mode=test dataset.test_mode=pose non_rigid=hashgrid_smplx rigid=skinning_field_smplx pose_correction=none dataset=ablation_study/smplx/x_humans_00036_take169



echo "Done."
echo FINISHED at $(date)