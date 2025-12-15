# General
import sys, os, json, argparse, torch, time
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

    device_map = {"": f"cuda:0"}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpts/Scone")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--input_image_paths", type=str, nargs='+', required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = f"outputs/seed_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    inferencer = load_model(args)

    # Prepare Inputs for Inference
    input_list = [Image.open(image_path) for image_path in args.input_image_paths] + [args.instruction]
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # Print Prompt and Process Image
    print(args.instruction)
    print('-' * 10)
    img = get_image(inferencer, input_list)

    # Clean up prompt for saving
    prompt = args.instruction.replace(" ", "_").replace(".", "_")
    prompt = prompt[:60] if len(prompt) >= 60 else prompt
    save_path = f"{save_dir}/{timestamp}_{prompt}.png"
    img.save(save_path)
    print(f'Image saved as {save_path}')