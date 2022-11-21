# python ./scripts/inference_new.py \
# --images_dir=/content/drive/MyDrive/celeb/CelebAMask-HQ/CelebA-HQ-img --edit_attribute='inversion' \
# --save_dir=/content/output  /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/inference_cars.py \
# --images_dir=/content/drive/MyDrive/car_ims --edit_attribute='inversion' --n_sample 100 \
# --save_dir=/content/drive/MyDrive/HFGI_New/HFGI/hfgi_output/cars  /content/drive/MyDrive/HFGI_New/HFGI/experiment/cars_encode/checkpoints/iteration_40000.pt

python ./scripts/edit.py \
--images_dir=/content/test  --edit_attribute='smile' --edit_degree=1.0 \
--save_dir=/content/output  /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=/content/drive/MyDrive/celeb/CelebAMask-HQ/CelebA-HQ-img  --edit_attribute='age' --edit_degree=-2.0 \
# --save_dir=/content/drive/MyDrive/HFGI_New/HFGI/hfgi_output/age_negative  /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=/content/drive/MyDrive/celeb/CelebAMask-HQ/CelebA-HQ-img --edit_attribute='lip'  \
# --save_dir=/content/drive/MyDrive/HFGI_New/HFGI/hfgi_output/lip    /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=/content/drive/MyDrive/celeb/CelebAMask-HQ/CelebA-HQ-img --edit_attribute='beard'  \
# --save_dir=/content/drive/MyDrive/HFGI_New/HFGI/hfgi_output/beard    /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=/content/drive/MyDrive/celeb/CelebAMask-HQ/CelebA-HQ-img --edit_attribute='eyes'  \
# --save_dir=/content/drive/MyDrive/HFGI_New/HFGI/hfgi_output/eyes    /content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_condition_first/checkpoints/iteration_16000.pt

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 
