################## Faces #############################

python ./scripts/train.py   --dataset_type='ffhq_frontalization'  --start_from_latent_avg \
--id_lambda=1.1  --val_interval=200000 --save_interval=10000 --max_steps=100000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0.5  \
--stylegan_weights='./pretrained/stylegan2-ffhq-config-f.pt' --checkpoint_path "/content/drive/MyDrive/HFGI_New/HFGI/experiment/ffhq_frontalization/checkpoints/iteration_20000.pt" \
--workers=48  --batch_size=16  --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/ffhq_frontalization' \
--lpips_lambda 0.1 --l2_lambda 0.2


# python ./scripts/train.py   --dataset_type='cars_encode'  --start_from_latent_avg \
# --id_lambda=0.1  --val_interval=200000 --save_interval=10000 --max_steps=100000  --stylegan_size=512 --is_train=True \
# --distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0.1  \
# --stylegan_weights='/content/drive/MyDrive/HFGI_New/HFGI/pretrained/stylegan2-car-config-f.pt' --checkpoint_path='/content/drive/MyDrive/HFGI/HFGI/experiment/cars_encode/checkpoints/iteration_90000.pt'  \
# --workers=48  --batch_size=4  --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/cars_encode' 