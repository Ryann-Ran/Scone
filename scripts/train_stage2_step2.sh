num_nodes=1
node_rank=0
master_addr='localhost'
master_port=29500
model_path=./ckpts/Scone_stage2_step1/0001000
EXP_NAME=Scone_stage2_step2
GPU_NUM=8

torchrun \
    --nnodes=$num_nodes \
    --node_rank=$node_rank \
    --nproc_per_node=$GPU_NUM \
    --master_addr=$master_addr \
    --master_port=$master_port \
    train/pretrain_unified_navit.py \
    --dataset_config_file ./data/configs/scone_stage2.yaml \
    --model_path $model_path \
    --layer_module Qwen2MoTDecoderLayer \
    --max_latent_size 64 \
    --resume-from $model_path \
    --finetune_from_hf True \
    --auto_resume True \
    --resume-model-only True \
    --finetune-from-ema False \
    --log_every 1 \
    --lr 2e-5 \
    --num_worker 3 \
    --num_shard $GPU_NUM \
    --wandb_name "${EXP_NAME}" \
    --wandb_runid 0 \
    --results_dir "./results/${EXP_NAME}" \
    --checkpoint_dir "./ckpts/${EXP_NAME}" \
    --cpu_offload False \
    --save_every 1000 \
    --expected_num_tokens 26496 \
    --max_num_tokens 26496 \
    --max_num_tokens_per_sample 26496 \
    --visual_und True \
    --freeze_und False \
    --freeze_vit True \
    --freeze_llm False \
    --freeze_llm_und True \
    --freeze_llm_gen True \
    --freeze_mlp_und True \
    --freeze_mlp_gen True \
    --text_cond_dropout_prob 0.1 \
    --vae_cond_dropout_prob 0.0 \
    --vit_cond_dropout_prob 0.0
