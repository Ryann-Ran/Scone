import os
import math
import json
import glob
import argparse
from collections import defaultdict
import numpy as np


def analyze_scores(json_lines):
    group_prompt_following_scores = {}
    group_subject_consistency_scores = {}
    group_overall_scores = {}

    group_dis_acc_scores = {}
    group_dis_prec_scores = {}
    group_dis_recall_scores = {}
    group_dis_f1_scores = {}
    group_dis_overall_scores = {}

    for task_type in json_lines.keys():
        prompt_following_scores = []
        subject_consistency_scores = []
        overall_scores = []

        dis_acc_scores = []
        dis_prec_scores = []
        dis_recall_scores = []
        dis_f1_scores = []
        dis_overall_scores = []

        for json_line in json_lines[task_type]:

            # COM
            prompt_following_score = json_line['PF_score']
            subject_consistency_score = json_line['SC_score']
            overall_score = math.sqrt(prompt_following_score * subject_consistency_score)

            prompt_following_scores.append(prompt_following_score)
            subject_consistency_scores.append(subject_consistency_score)
            overall_scores.append(overall_score)

            # DIS
            if "distinction" in task_type:
                dis_acc_score = json_line['DIS_acc']
                dis_prec_score = json_line['DIS_prec']
                dis_recall_score = json_line['DIS_recall']
                dis_f1_score = json_line['DIS_f1']
                dis_overall_score = round((dis_acc_score + dis_f1_score) / 2, 3)

                dis_acc_scores.append(dis_acc_score)
                dis_prec_scores.append(dis_prec_score)
                dis_recall_scores.append(dis_recall_score)
                dis_f1_scores.append(dis_f1_score)
                dis_overall_scores.append(dis_overall_score)

        group_prompt_following_scores[task_type] = np.mean(prompt_following_scores)
        group_subject_consistency_scores[task_type] = np.mean(subject_consistency_scores)
        group_overall_scores[task_type] = np.mean(overall_scores)

        if "distinction" in task_type:
            group_dis_acc_scores[task_type] = np.mean(dis_acc_scores)
            group_dis_prec_scores[task_type] = np.mean(dis_prec_scores)
            group_dis_recall_scores[task_type] = np.mean(dis_recall_scores)
            group_dis_f1_scores[task_type] = np.mean(dis_f1_scores)
            group_dis_overall_scores[task_type] = np.mean(dis_overall_scores)

    return (
        group_prompt_following_scores,
        group_subject_consistency_scores,
        group_overall_scores,
        group_dis_acc_scores,
        group_dis_prec_scores,
        group_dis_recall_scores,
        group_dis_f1_scores,
        group_dis_overall_scores
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)  # "omnicontext" or "sconeeval"
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    save_score_try_dir_list = ["score_try1", "score_try2", "score_try3"]

    for save_score_try_dir in save_score_try_dir_list:
        result_json_files = glob.glob(
            os.path.join(args.result_dir, args.model_name, save_score_try_dir, "**/*.jsonl")
        )
        print(f"{len(result_json_files)=}")
        print(f"{result_json_files=}")

        result_json_lines = defaultdict(list)
        for result_json_file in result_json_files:
            with open(result_json_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    task_type = os.path.basename(os.path.dirname(result_json_file))
                    result_json_lines[task_type].append(data)

        (
            group_prompt_following_scores,
            group_subject_consistency_scores,
            group_overall_scores,
            group_dis_acc_scores,
            group_dis_prec_scores,
            group_dis_recall_scores,
            group_dis_f1_scores,
            group_dis_overall_scores
        ) = analyze_scores(result_json_lines)

        # Specify output order
        ordered_task_types_omnicontext = [
            "single_character", "single_object",
            "multi_character", "multi_object", "multi_character_object",
            "scene_character", "scene_object", "scene_character_object"
        ]
        ordered_task_types_sconeeval = [
            "composition_single", "composition_multi",
            "distinction_single_cross", "distinction_single_intra",
            "distinction_multi_cross", "distinction_multi_intra"
        ]

        # Check if all keys are included
        if all(k in group_prompt_following_scores.keys() for k in ordered_task_types_omnicontext):
            task_type_order = ordered_task_types_omnicontext
        elif all(k in group_prompt_following_scores.keys() for k in ordered_task_types_sconeeval):
            task_type_order = ordered_task_types_sconeeval
        else:
            task_type_order = list(group_prompt_following_scores.keys())
            print(f"{task_type_order=}")
            if args.benchmark == "omnicontext":
                raise Exception("Task types in OmniContext benchmark is missing.")
                # pass
            if args.benchmark == "sconeeval":
                raise Exception("Task types in SconeEval benchmark is missing.")
                # pass

        if args.benchmark == "omnicontext":
            # Print in order
            for task_type in task_type_order:
                print(f"{task_type}: {group_prompt_following_scores[task_type]:.3f}, "
                        f"{group_subject_consistency_scores[task_type]:.3f}, "
                        f"{group_overall_scores[task_type]:.3f}")

            print(f"Average: {np.mean(list(group_prompt_following_scores.values())):.3f}, "
                    f"{np.mean(list(group_subject_consistency_scores.values())):.3f}, "
                    f"{np.mean(list(group_overall_scores.values())):.3f}")

            # Write results to file
            save_dir = os.path.join(args.result_dir, args.model_name, save_score_try_dir)
            output_path = os.path.join(save_dir, "scores.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for task_type in task_type_order:
                    f.write(
                        f"{task_type}: "
                        f"{group_prompt_following_scores[task_type]:.3f}, "
                        f"{group_subject_consistency_scores[task_type]:.3f}, "
                        f"{group_overall_scores[task_type]:.3f}\n"
                    )
                f.write(
                    f"Average: "
                    f"{np.mean(list(group_prompt_following_scores.values())):.3f}, "
                    f"{np.mean(list(group_subject_consistency_scores.values())):.3f}, "
                    f"{np.mean(list(group_overall_scores.values())):.3f}\n"
                )
            print(f"Results written to: {output_path}")

        elif args.benchmark == "sconeeval":
            # Print in order
            for task_type in task_type_order:
                if "distinction" in task_type:
                    print(f"{task_type}: {group_prompt_following_scores[task_type]:.3f}, "
                            f"{group_subject_consistency_scores[task_type]:.3f}, "
                            f"{group_overall_scores[task_type]:.3f}, "
                            f"{group_dis_acc_scores[task_type]:.3f}, "
                            f"{group_dis_prec_scores[task_type]:.3f}, "
                            f"{group_dis_recall_scores[task_type]:.3f}, "
                            f"{group_dis_f1_scores[task_type]:.3f}, "
                            f"{group_dis_overall_scores[task_type]:.3f}")

                else:
                    print(f"{task_type}: {group_prompt_following_scores[task_type]:.3f}, "
                            f"{group_subject_consistency_scores[task_type]:.3f}, "
                            f"{group_overall_scores[task_type]:.3f}")

            print(f"Average: {np.mean(list(group_prompt_following_scores.values())):.3f}, "
                    f"{np.mean(list(group_subject_consistency_scores.values())):.3f}, "
                    f"{np.mean(list(group_overall_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_acc_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_prec_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_recall_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_f1_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_overall_scores.values())):.3f}")

            # Write results to file
            save_dir = os.path.join(args.result_dir, args.model_name, save_score_try_dir)
            output_path = os.path.join(save_dir, "scores.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                for task_type in task_type_order:
                    if "distinction" in task_type:
                        f.write(
                            f"{task_type}: "
                            f"{group_prompt_following_scores[task_type]:.3f}, "
                            f"{group_subject_consistency_scores[task_type]:.3f}, "
                            f"{group_overall_scores[task_type]:.3f}, \t"
                            f"{group_dis_acc_scores[task_type]:.3f}, "
                            f"{group_dis_prec_scores[task_type]:.3f}, "
                            f"{group_dis_recall_scores[task_type]:.3f}, "
                            f"{group_dis_f1_scores[task_type]:.3f}, "
                            f"{group_dis_overall_scores[task_type]:.3f}\n"
                        )
                    else:
                        f.write(
                            f"{task_type}: "
                            f"{group_prompt_following_scores[task_type]:.3f}, "
                            f"{group_subject_consistency_scores[task_type]:.3f}, "
                            f"{group_overall_scores[task_type]:.3f}\n"
                        )
                f.write(
                    f"Average: "
                    f"{np.mean(list(group_prompt_following_scores.values())):.3f}, "
                    f"{np.mean(list(group_subject_consistency_scores.values())):.3f}, "
                    f"{np.mean(list(group_overall_scores.values())):.3f}, \t"
                    f"{np.mean(list(group_dis_acc_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_prec_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_recall_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_f1_scores.values())):.3f}, "
                    f"{np.mean(list(group_dis_overall_scores.values())):.3f}"
                )
            print(f"Results written to: {output_path}")