import os
import copy
import argparse
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

from bench_score import BenchScore

def process_single_item(item, vie_score, max_retries=5, test_image_dir=None, benchmark="sconeeval"):
    result_dict = copy.deepcopy(item)
    
    # Do not store PIL objects in the result dictionary
    if "input_images" in result_dict.keys():
        if not isinstance(result_dict["input_images"][0], str):
            del result_dict['input_images']

    instruction = item['instruction']
    key = item['key']
    
    input_images = item.get('input_images')
    if isinstance(input_images[0], str):
        input_images = [
            Image.open(os.path.join(test_image_dir, img)).convert("RGB") 
            for img in input_images
        ]
    output_image = Image.open(item['output_image_path']).convert("RGB")

    ori_img_sizes = [input_image.size for input_image in input_images]
    new_img_sizes = []

    # Calculate new image sizes
    for ori_img_size in ori_img_sizes:
        if ori_img_size[0] * ori_img_size[1] > 1024 * 1024:
            ratio = math.sqrt(1024 * 1024 / (ori_img_size[0] * ori_img_size[1]))
            new_img_size = (int(ori_img_size[0] * ratio), int(ori_img_size[1] * ratio))
        else:
            new_img_size = ori_img_size
        
        new_img_size = (new_img_size[0] // 16 * 16, new_img_size[1] // 16 * 16)
        new_img_sizes.append(new_img_size)

    input_images = [
        input_image.resize(new_img_size) 
        for input_image, new_img_size in zip(input_images, new_img_sizes)
    ]

    if item['task_type'].find('scene') != -1 or "S" in item.get("case_type", ""):
        with_scene = True
    else:
        with_scene = False

    if benchmark == "sconeeval":
        sconeeval_flag = True
        if "distinction" in item['task_type']:
            subject_list = item['subject_list']
            DIS_gt = item['subject_label']
        else:
            subject_list = None
            DIS_gt = None
    elif benchmark == "omnicontext":
        sconeeval_flag = False
        subject_list = None
        DIS_gt = None
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    score_dict = vie_score.evaluate(
        input_images + [output_image], 
        instruction, 
        with_scene=with_scene, 
        subject_list=subject_list, 
        DIS_gt=DIS_gt, 
        sconeeval_flag=sconeeval_flag
    )
    
    print(f"{score_dict=}", flush=True)

    result_dict['PF_score'] = score_dict['PF_scores']['score']
    result_dict['PF_score_reason'] = score_dict['PF_scores']['reasoning']
    result_dict['SC_score'] = score_dict['SC_scores']['score']
    result_dict['SC_score_reason'] = score_dict['SC_scores']['reasoning']

    if "distinction" in item['task_type']:
        for key, value in score_dict.items():
            if 'DIS' in key:
                result_dict[key] = value

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)  # "omnicontext" or "sconeeval"
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=100)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--test_image_dir", type=str, required=True)
    parser.add_argument("--openai_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--openai_key", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./eval/s2i/sconeeval/results")
    args = parser.parse_args()

    bench_score = BenchScore(args.openai_url, args.openai_key)

    # Load test data
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_dataset = [json.loads(line) for line in f]

    sub_datasets = defaultdict(list)
    for example in test_dataset:
        task_type = example['task_type']
        sub_datasets[task_type].append(example)
    print(f"{len(sub_datasets)=}")

    # Automatically select save subdirectory score_tryX
    save_root = os.path.join(args.save_dir, args.model_name)

    save_score_try_dir = None
    for i in range(1, 4):
        cand = f"score_try{i}"
        save_score_try_dir_cand = os.path.join(save_root, cand)
        if os.path.exists(save_score_try_dir_cand):
            folder_count = sum(os.path.isdir(os.path.join(save_score_try_dir_cand, name)) for name in os.listdir(save_score_try_dir_cand))
            print(f"folder_count={folder_count}")
        else:
            folder_count = 0

        if not os.path.exists(save_score_try_dir_cand) or folder_count < len(sub_datasets):
            save_score_try_dir = cand
            break
            
    if save_score_try_dir is None:
        # If 1..3 all exist, raise error indicating scoring has been done 3 times
        raise Exception("score_tryX all exist.")

    all_result_list = []
    
    for task_type, sub_data in sub_datasets.items():
        result_list = []
        json_path = os.path.join(save_root, save_score_try_dir, task_type, "score.jsonl")

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    result_list.append(result)
            print(f"Loaded {json_path} for {task_type}, length: {len(result_list)}")
            continue

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for item in sub_data:
                key = item["key"]
                output_image_path = os.path.join(args.result_dir, args.model_name, task_type, f"{key}.png")
                item['output_image_path'] = output_image_path
                
                assert os.path.exists(output_image_path), f"{output_image_path} not exists."
                # if not os.path.exists(output_image_path):
                #     continue

                future = executor.submit(
                    process_single_item, 
                    item, 
                    bench_score, 
                    test_image_dir=args.test_image_dir, 
                    benchmark=args.benchmark
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), unit="image", desc=f"Processing {task_type}"):
                result = future.result()
                if result:
                    result_list.append(result)

        all_result_list.extend(result_list)
                
        # Save group-specific result
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            for result in result_list:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Saved {json_path} for {task_type}, length: {len(result_list)}")

    combined_json_path = os.path.join(save_root, save_score_try_dir, "combined_score.jsonl")

    os.makedirs(os.path.dirname(combined_json_path), exist_ok=True)
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        for result in all_result_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
