CUDA_VISIBLE_DEVICES=0 python inference_single_case.py \
    --model_path ./ckpts/Scone \
    --instruction "The man in image 2 is holding the object with a blue-and-red top image 1." \
    --input_image_paths ./test_images/image_object.png ./test_images/image_character.png \
    --seed 1234 \
