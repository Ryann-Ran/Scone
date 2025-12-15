export CUDA_VISIBLE_DEVICES=1,2,3,4

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    inference_benchmark.py \
    --model_path ./ckpts/Scone \
    --metadata_file ../OmniContext-jsonl/data.jsonl \
    --image_dir ../OmniContext-jsonl \
    --seed 1234 \
    --output_dir ./results/omnicontext/Scone
    
