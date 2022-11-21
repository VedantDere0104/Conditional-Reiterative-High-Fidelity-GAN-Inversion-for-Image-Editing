import argparse
import torch
import numpy as np
import sys
import os

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
from editings import latent_editor
import time
resize_dims=(256,256)

def display_alongside_source_image(result_image, source_image):
    res = np.concatenate([np.array(source_image.resize(resize_dims)),
                          np.array(result_image.resize(resize_dims))], axis=1)
    return Image.fromarray(res)

def main(args):
    net, opts = setup_model(args.ckpt, device)
    edit_directory_path = '/content/drive/MyDrive/HFGI_New/HFGI/metrics/celeb_hq/inverted'
    original_directory_path = '/content/drive/MyDrive/HFGI_New/HFGI/metrics/celeb_hq/original'
    os.makedirs(edit_directory_path, exist_ok=True)
    args, data_loader = setup_data_loader(args, opts)
    for i , batch in enumerate(data_loader):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break        

        transformed_image = batch.to(device).float()

        with torch.no_grad():
            x = transformed_image.cuda()
            

            latent_codes_ = get_latents(net, x , True)
            
            # calculate the distortion map
            tic = time.time()
            imgs_, _ = net.decoder([latent_codes_[0].unsqueeze(0).cuda()],None, input_is_latent=True, randomize_noise=False, return_latents=True)
            # tensor2im(torch.nn.functional.interpolate(torch.clamp(imgs_, -1., 1.), size=(256,256) , mode='bilinear')[0]).save('/content/output/first_decoder_1.png')
            res_ = x -  torch.nn.functional.interpolate(torch.clamp(imgs_, -1., 1.), size=(256,256) , mode='bilinear')

            # ADA
            img_edit = torch.nn.functional.interpolate(torch.clamp(imgs_, -1., 1.), size=(256,256) , mode='bilinear')
            res_align_  = net.grid_align(torch.cat((res_, img_edit  ), 1))

            # consultation fusion
            conditions_ = net.residue(res_align_)
            # tensor2im(res_[0]).save('/content/output/conduction_1.png')

            result_image, _ = net.decoder([latent_codes_],conditions_, input_is_latent=True, randomize_noise=False, return_latents=True)
            result_image = torch.nn.functional.interpolate(result_image, size=(256,256) , mode='bilinear')
            # tensor2im(result_image[0]).save('/content/output/first_decoder_2.png')

            imgs = result_image
            res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
            # tensor2im(res[0]).save('/content/output/conduction_2.png')

            # ADA
            img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
            res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

            # consultation fusion
            conditions = net.residue(res_align)

            # latent_codes = get_latents(net, result_image)
            
            # calculate the distortion map
            imgs, _ = net.decoder([latent_codes_[0].unsqueeze(0).cuda()],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
            imgs = result_image
            # tensor2im(torch.nn.functional.interpolate(torch.clamp(result_image, -1., 1.), size=(256,256) , mode='bilinear')[0]).save('/content/output/first_decoder_3.png')
            res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
            # tensor2im(res[0]).save('/content/output/conduction_3.png')

            # ADA
            img_edit = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')
            res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

            # consultation fusion
            conditions = net.residue(res_align)
            

            result_image, latent_codes = net.decoder([latent_codes_],conditions_, input_is_latent=True, randomize_noise=False, return_latents=True)

            # tensor2im(torch.nn.functional.interpolate(torch.clamp(result_image, -1., 1.), size=(256,256) , mode='bilinear')[0]).save('/content/output/first_decoder_4.png')


            toc = time.time()
            print(toc-tic)
            imgs = torch.nn.functional.interpolate(result_image, size=(256,256) , mode='bilinear')
            #result = tensor2im(imgs[0])

            img = display_alongside_source_image(tensor2im(imgs[0]), tensor2im(x[0]))
            # im_save_path_inv = os.path.join(edit_directory_path, f"{i:05d}.jpg")
            # im_save_path_org = os.path.join(original_directory_path, f"{i:05d}.jpg")
            saving_dir = os.path.join(args.save_dir, f"{i:05d}.jpg")
            Image.fromarray(np.array(img)).save(saving_dir)            
            # tensor2im(imgs[0]).save(im_save_path_inv)
            # tensor2im(x[0]).save(im_save_path_org)

def setup_data_loader(args, opts):
    opts.dataset_type = 'cars_encode'
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)