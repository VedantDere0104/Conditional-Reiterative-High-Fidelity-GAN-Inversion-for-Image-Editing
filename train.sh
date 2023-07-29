################## Faces #############################

python ./scripts/train.py   --dataset_type='ffhq_encode'  --start_from_latent_avg \
--val_interval=200000 --save_interval=10000 --max_steps=100000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9  \
--stylegan_weights='./pretrained/stylegan2-ffhq-config-f.pt' --checkpoint_path "ckpt" \
--workers=48  --batch_size=16  --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/ffhq' \


# python ./scripts/train.py   --dataset_type='cars_encode'  --start_from_latent_avg \
#  --val_interval=200000 --save_interval=10000 --max_steps=100000  --stylegan_size=512 --is_train=True \
# --distortion_scale=0.15 --aug_rate=0.9  \
# --stylegan_weights='./pretrained/stylegan2-car-config-f.pt' --checkpoint_path='ckpt_cars'  \
# --workers=48  --batch_size=4  --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/cars_encode' 
