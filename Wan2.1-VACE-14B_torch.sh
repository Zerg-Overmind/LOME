export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # ← key line

GPUS_PER_NODE=$((8 * $NUM_NODES))
MASTER_PORT=29500
NODE_RANK=${SLURM_NODEID:-0}
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}

MASTER_ADDR=$(
python - <<'PY'
import os, re
nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
if '[' not in nodelist:
    print(nodelist.split(',')[0]); exit()
prefix, inner = re.match(r'([^[]+)\[(.+)\]', nodelist).groups()
first = inner.split(',')[0]
if '-' in first:
    s,e = first.split('-'); print(f"{prefix}{int(s):0{len(s)}d}")
else:
    print(prefix + first)
PY
) || MASTER_ADDR=$(hostname)

echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES"


# Launch the training script using torchrun for direct control.
# We now use the --rdzv-endpoint argument with our dynamic variables.
accelerate launch \
    --num_processes ${GPUS_PER_NODE} \
    --num_machines ${NUM_NODES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path yours/datasets/ego-dex \
    --dataset_metadata_path yours/datasets/ego-dex/metadata_vace.csv \
    --data_file_keys "video,vace_reference_image,vace_video,prompt,pose" \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
    --learning_rate 5e-6 \
    --num_epochs 100 \
    --remove_prefix_in_ckpt "pipe.vace." \
    --output_path "./models/train/Wan2.1-VACE-14B_full" \
    --trainable_models "vace" \
    --lora_base_model "vace" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 128 \
    --extra_inputs "vace_reference_image,vace_video,prompt,pose" \
    --save_steps 80 \
    --use_gradient_checkpointing_offload



