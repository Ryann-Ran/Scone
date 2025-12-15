# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, S2IIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    's2i': S2IIterableDataset
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    's2i': {
        'scone_single_candidate_base': {
            'data_dir': '/your_data_path/Scone-S2I-57K/parquet_data/scone_single_candidate_base',
            'num_files': 70,
            'num_total_samples': 70000, # 70 K
            "parquet_info_path": '/your_data_path/Scone-S2I-57K/parquet_info/scone_single_candidate_base.json', # information of the parquet files
        },
        'scone_single_candidate_refined': {
            'data_dir': '/your_data_path/Scone-S2I-57K/parquet_data/scone_single_candidate_refined',
            'num_files': 22,
            'num_total_samples': 21728, # 22 K
            "parquet_info_path": '/your_data_path/Scone-S2I-57K/parquet_info/scone_single_candidate_refined.json', # information of the parquet files
        },
        'scone_multi_candidate': {
            'data_dir': '/your_data_path/Scone-S2I-57K/parquet_data/scone_multi_candidate',
            'num_files': 36,
            'num_total_samples': 35023, # 35 K
            "parquet_info_path": '/your_data_path/Scone-S2I-57K/parquet_info/scone_multi_candidate.json', # information of the parquet files
        },
    }
}