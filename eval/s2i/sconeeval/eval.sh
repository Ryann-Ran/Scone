MODEL_NAME="Scone"

python ./eval/s2i/sconeeval/test_score.py \
    --benchmark sconeeval \
    --result_dir ./eval_results/sconeeval \
    --model_name ${MODEL_NAME} \
    --max_workers 1 \
    --test_data ../SconeEval/data.jsonl \
    --test_image_dir ../SconeEval \
    --openai_key <Your-API-Key> \
    --save_dir ./eval/s2i/sconeeval/results

python ./eval/s2i/sconeeval/calculate_statistics.py \
    --benchmark sconeeval \
    --result_dir ./eval/s2i/sconeeval/results \
    --model_name ${MODEL_NAME}