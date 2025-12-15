CUDA_VISIBLE_DEVICES=0 python inference_single_case.py \
    --model_path ./ckpts/Scone \
    --instruction "The man from image 2 holds the object which has a blue-and-red top in image 1 in a coffee shop." \
    --input_image_paths ./test_images/image_object.png ./test_images/image_character.png \
    --seed 1234 \
