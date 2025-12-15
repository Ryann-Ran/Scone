MODEL_NAME="Scone"

python ./eval/s2i/sconeeval/test_score.py \
    --benchmark omnicontext \
    --result_dir ./eval_results/omnicontext \
    --model_name ${MODEL_NAME} \
    --max_workers 1 \
    --test_data ../OmniContext-jsonl/data.jsonl \
    --test_image_dir ../OmniContext-jsonl \
    --openai_key <Your-API-Key> \
    --save_dir ./eval/s2i/omnicontext/results

python ./eval/s2i/sconeeval/calculate_statistics.py \
    --benchmark omnicontext \
    --result_dir ./eval/s2i/omnicontext/results \
    --model_name ${MODEL_NAME}