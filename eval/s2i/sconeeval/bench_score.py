import time
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)  # for SconeEval Benchmark
from prompt_generator import PromptGenerator
from openai_util import ask_gpt4o
from json_util import mllm_output_to_dict, mllm_output_to_int


def compute_classification_metrics(gt, prediction):
    """
    Args:
        gt: list of list of int, e.g. [[0, 1], [0]]
        prediction: list of list of int, same format as gt
    Returns:
        dict: containing accuracy, precision, recall, f1, and overall
    """
    # Flatten the lists
    gt_flat = [v for sublist in gt for v in sublist]
    pred_flat = [v for sublist in prediction for v in sublist]

    if len(gt_flat) != len(pred_flat):
        raise ValueError("Lengths of gt and prediction do not match")

    DIS_acc = accuracy_score(gt_flat, pred_flat) * 10
    DIS_prec = precision_score(gt_flat, pred_flat, zero_division=0) * 10
    DIS_recall = recall_score(gt_flat, pred_flat, zero_division=0) * 10
    DIS_f1 = f1_score(gt_flat, pred_flat, zero_division=0) * 10

    metrics = {
        "DIS_acc": DIS_acc,
        "DIS_prec": DIS_prec,
        "DIS_recall": DIS_recall,
        "DIS_f1": DIS_f1,
        "DIS_overall": round((DIS_acc + DIS_f1) / 2, 3),
        "DIS_pred": prediction
    }
    return metrics


class BenchScore:
    def __init__(self, openai_url: str, openai_key: str) -> None:
        self.openai_url = openai_url
        self.openai_key = openai_key
        self.prompt_generator = PromptGenerator()

    def evaluate(self, input_image_paths, instruction, with_scene=False, 
                    distinction_flag=False, sconeeval_flag=False, 
                    subject_list=None, DIS_gt=None):
        results_dict = {}

        max_tries = 3
        PF_scores = None
        SC_scores = None
        if subject_list is not None:
            DIS_scores = None

        for try_idx in range(max_tries):
            try:
                PF_prompt = self.prompt_generator(
                    instruction, 
                    task_type="prompt_following", 
                    sconeeval_flag=sconeeval_flag
                )
                SC_prompt = self.prompt_generator(
                    instruction, 
                    task_type="subject_consistency", 
                    with_scene=with_scene, 
                    sconeeval_flag=sconeeval_flag
                )

                PF_results = ask_gpt4o(input_image_paths, PF_prompt, self.openai_url, self.openai_key)
                SC_results = ask_gpt4o(input_image_paths, SC_prompt, self.openai_url, self.openai_key)

                PF_scores = mllm_output_to_dict(PF_results)
                SC_scores = mllm_output_to_dict(SC_results)

                if PF_scores == "rate_limit_exceeded" or SC_scores == "rate_limit_exceeded":
                    raise Exception("rate_limit_exceeded")

                if subject_list is not None:
                    DIS_preds = []
                    # Iterate through all reference images
                    for idx, sub_list in enumerate(subject_list):
                        DIS_preds_per_image = []
                        input_image_paths_per_image = [input_image_paths[idx], input_image_paths[-1]]
                        
                        # Iterate through all subjects in the current reference image
                        for subject in sub_list:
                            # instruction is actually unused here
                            DIS_prompt = self.prompt_generator(instruction, task_type="distinction", subject=subject)
                            # Results returned by GPT
                            DIS_results = ask_gpt4o(input_image_paths_per_image, DIS_prompt, self.openai_url, self.openai_key)
                            # Convert GPT result to integer
                            DIS_pred = mllm_output_to_int(DIS_results)
                            
                            if DIS_pred not in [0, 1]:
                                raise Exception(f"{DIS_pred} not in 0, 1.")
                            DIS_preds_per_image.append(DIS_pred)
                        DIS_preds.append(DIS_preds_per_image)

                    if not len(DIS_preds) == len(subject_list):
                        raise Exception("The length of DIS_preds does not correspond to subject_list.")
                    
                    DIS_scores = compute_classification_metrics(DIS_gt, DIS_preds)
                
                # If successful, break the retry loop
                break

            except Exception as e:
                backoff_time = min(2 ** try_idx, 300)
                print(f"{e}, {instruction=}, Attempt {try_idx+1} failed, retrying after {backoff_time} seconds...")
                time.sleep(backoff_time)

        assert PF_scores is not None and SC_scores is not None
        results_dict["PF_scores"] = PF_scores
        results_dict["SC_scores"] = SC_scores

        if subject_list is not None:
            assert DIS_scores is not None
            results_dict.update(DIS_scores)

        return results_dict
