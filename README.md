 # Conditional Reiterative High-Fidelity GAN Inversion for Image Editing


>Our work introduces a conditional reiteration mechanism for high-fidelity GAN inversion, preserving image-specific details for both normal and out-of-domain images (e.g., >heavy makeup faces). The HFGI encoder's single-stage conditional latent maps result in blurry regions in restored images and loss of detailed information during editing. To >address this, we propose a reiterative conditional latents method that restores image-specific details sharply without changing HFGI's architecture. The process involves >two stages of iterations, reconstructing the image in the first stage, and refining image-specific details using conditional latent codes in the second stage. Our model >successfully inverts out-of-domain images while preserving all details and supports InterfaceGAN, GANspace, and StyleClip for editing. We also compare our approach with >state-of-the-art GAN inversion methods on FFHQ, demonstrating significant improvements in inversion and editing quality.

![Screenshot from 2023-07-29 20-03-30](https://github.com/VedantDere0104/High-Fidelity-GAN-Inversion-with-Condition-Reiteration-for-image-editing/assets/76057253/445f3cbe-021c-46aa-93ec-3d602e7e5327)

# Setup 

```
git clone https://github.com/VedantDere0104/High-Fidelity-GAN-Inversion-with-Condition-Reiteration-for-image-editing.git
cd High-Fidelity-GAN-Inversion-with-Condition-Reiteration-for-image-editing
```

# Requirements 

```
pip install -r requirements.txt
pip install --upgrade gdown 
```

# Download Model
## Human Faces
```
gdown --fuzzy "https://drive.google.com/file/d/1OMXlZabcq2M44NYz9HlqHanUcely8ZUb/view?usp=sharing"
```
## cars 
```
gdown --fuzzy "https://drive.google.com/file/d/1R9tJ0koXpkDzlmErsb-TvuAddAeqBwgZ/view?usp=sharing"
```


# Inference 
First, create a folder to store the testing images. Ensure that the images in this folder are properly aligned before proceeding with the testing.

```
bash inference.sh
```
You can check the commented code in `inference.sh` file which shows different edits. Uncomment the necessary edit and run.
For cars inversion please uncomment `inference_cars.py` line in bash file.
## Arguments 

| Argument | working |
|----------|------   |
| --images_dir | input image directory |
| --edit_attribute | editing attribute use inversion to invert image. Currently supported attributes are `smile`, `age`, `lip`, `beard`, `eyes` |
| --edit_degree | edit degree required to `smile` and `age` |
| --save_dir | output saving directory |
| ckpt | Models checkpoint |


# Training

## Required Models

| Model | Description
| :--- | :----------
|[StyleGAN2 (FFHQ)](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | Pretrained face generator on FFHQ  from [rosinality](https://github.com/rosinality/stylegan2-pytorch).
|[e4e (FFHQ)](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) | Pretrained initial encoder on FFHQ  from [omertov](https://github.com/omertov/encoder4editing).
|[Feature extractor (for face)](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for ID loss calculation.
|[Feature extractor (for car)](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view?usp=sharing) | Pretrained ResNet-50 model taken from [omertov](https://github.com/omertov/encoder4editing) for ID loss calculation.

Download these models and put in `\pretrained_models` directory.
Use below command to download the models
```
cd pretrained_models
gdown --fuzzy "model URL"
```

```
bash train.sh
```


## Acknowledgement
Thanks to [e4e](https://github.com/omertov/encoder4editing) and [HFGI](https://github.com/Tengfei-Wang/HFGI/tree/main) for sharing their code.
