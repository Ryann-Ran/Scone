# General
import sys, os, json, argparse, torch
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from datetime import timedelta

# Model-specific
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from safetensors.torch import load_file
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from inferencer import InterleaveInferencer
from data.transforms import ImageTransform


def load_model(args):
    model_path = args.model_path

    # Load LLM configuration
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # Load Vision Transformer (ViT) configuration
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # Load Autoencoder (VAE) model
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Configure model
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Load tokenizer and add special tokens
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    device_map = {"": f"cuda:{int(os.environ['LOCAL_RANK'])}"}
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}"

    # Load checkpoint and dispatch model
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(args.model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    ).eval()

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Create an inferencer for the model
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    return inferencer


def get_image(inferencer, input_list):
    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    output_dict = inferencer.interleave_inference(input_lists=input_list, **inference_hyper)
    return output_dict[0]


def filter_processed_samples(metadatas, output_dir):
    """
    Filter out already processed samples. If the image file exists and is valid, skip it.
    Otherwise, add the sample to the list.
    """
    remaining_metadatas = []

    for metadata in metadatas:
        task_type = metadata['task_type']
        key = metadata['key']
        task_dir = os.path.join(output_dir, task_type)
        os.makedirs(task_dir, exist_ok=True)

        # Build image file path
        image_file = os.path.join(task_dir, f"{key}.png")

        # Check if the image file exists and is valid
        if os.path.exists(image_file):
            try:
                # Try opening the image to verify its validity
                with Image.open(image_file) as img:
                    img.verify()  # Verify image integrity
                # If the file exists and is valid, skip the sample
            except (IOError, SyntaxError):
                # If the image is invalid, add it to the list
                remaining_metadatas.append(metadata)
        else:
            # If the image file does not exist, add it to the list
            remaining_metadatas.append(metadata)

    return remaining_metadatas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))

    rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    # Load and filter completed samples (i.e., samples with images already in the sample directory)
    if args.metadata_file.endswith('.json'):
        with open(args.metadata_file, "r", encoding="utf-8") as fp:
            metadatas = json.load(fp)
    elif args.metadata_file.endswith('.jsonl'):
        metadatas = []
        with open(args.metadata_file, "r", encoding="utf-8") as fp:
            for line in fp:
                metadatas.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported file format: {input_file_path}")

    # Filter out processed samples
    remaining_metadatas = filter_processed_samples(metadatas, output_dir)
    print(f"After filtering, {len(remaining_metadatas)} samples left to be processed.")
    if len(remaining_metadatas) == 0:
        sys.exit(0)

    # Allocate data for each GPU
    prompts_per_gpu = (len(remaining_metadatas) + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, len(remaining_metadatas))
    local_metadatas = remaining_metadatas[start:end]
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    # Load model
    inferencer = load_model(args)

    # Process each sample
    for local_idx, metadata in tqdm(enumerate(local_metadatas)):
        task_type = metadata["task_type"]
        task_dir = os.path.join(output_dir, task_type)
        os.makedirs(task_dir, exist_ok=True)

        key = metadata['key']
        prompt = metadata['instruction']
        image_paths = metadata.get('input_images') or metadata.get('input_image', [])
        # Ensure image_paths is a list, especially for single-subject data
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        input_list = []
        for image_path in image_paths:
            input_list.append(Image.open(os.path.join(args.image_dir, image_path)))
        input_list.append(prompt)
        print(f"GPU {rank} processing prompt {local_idx}/{end - start}: '{prompt}'")

        # Generate image
        img = get_image(inferencer, input_list)
        img.save(os.path.join(task_dir, f"{key}.png"))

    print(f"GPU {rank} has completed all tasks")
    dist.barrier(device_ids=[rank])
