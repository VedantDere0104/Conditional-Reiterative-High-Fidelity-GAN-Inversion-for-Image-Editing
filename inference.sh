python ./scripts/inference_new.py \
--images_dir=test_imgs --edit_attribute='inversion' \
--save_dir=output_dir  iteration_16000.pt

# python ./scripts/inference_cars.py \
# --images_dir=car_ims --edit_attribute='inversion' --n_sample 100 \
# --save_dir=outputs_cars  iteration_40000.pt

# python ./scripts/edit.py \
# --images_dir=test_imgs  --edit_attribute='smile' --edit_degree=1.0 \
# --save_dir=output_smile  iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=test_imgs  --edit_attribute='age' --edit_degree=-2.0 \
# --save_dir=output_smile_neg  iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=test_imgs --edit_attribute='lip'  \
# --save_dir=output_lip   iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=test_imgs --edit_attribute='beard'  \
# --save_dir=output_beard  iteration_16000.pt

# python ./scripts/edit.py \
# --images_dir=test_imgs --edit_attribute='eyes'  \
# --save_dir=output_eyes   iteration_16000.pt



