setup="
. yours/anaconda3/etc/profile.d/conda.sh;
conda activate yours/workspace/env/LOME_env;
cd yours/LOME/path;
export WANDB_API_KEY=f11032bbec5ef2474a0cb60e1c0dd3fcaa21f49e;
export TORCH_HOME=yours/cache/torch_home;
export HF_HOME=yours/cache/hf_home;
"

timestamp=$(date +%Y%m%d_%H%M%S)
job_name=ego_av_concat_14B_${timestamp}

cmd="${setup} bash Wan2.1-VACE-14B_torch.sh"
submit_job --nodes 8 \
        --more_srun_args="--ntasks-per-node=1 --gpus-per-node=8" \
        --account yours/slurm/account \
        --partition yours/slurm/account --tasks_per_node=1 \
        -n $job_name --duration 4 \
        --autoresume_uninstrumented \
        --exclusive \
        --mounts='yours/' \
        --dependency=singleton \
        --notimestamp \
        --image='yours/image.sqsh' \
        --command  "${cmd}"
