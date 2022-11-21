dataset_paths = {
	#  Face Datasets (FFHQ - train, CelebA-HQ - test)
	'ffhq': '/content/drive/MyDrive/gfp_inversion/GFPGAN/data',
	'ffhq_val': '/content/drive/MyDrive/gfp_inversion/GFPGAN/test_data',

	#  Cars Dataset (Stanford cars)
	'cars_train': '/content/drive/MyDrive/car_ims',
	'cars_val': '/content/drive/MyDrive/car_ims',
}

model_paths = {
	'stylegan_ffhq': './pretrained/stylegan2-ffhq-config-f.pt',
	'ir_se50': './pretrained/model_ir_se50.pth',
	'shape_predictor': './pretrained/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained/moco_v2_800ep_pretrain.pt'
}
edit_paths = {
	'age': 'editing/interfacegan_directions/age.pt',
	'smile': 'editing/interfacegan_directions/smile.pt',
	'pose': 'editing/interfacegan_directions/pose.pt',
	'cars': 'editing/ganspace_directions/cars_pca.pt',
	'styleclip': {
		'delta_i_c': '/content/drive/MyDrive/HFGI_New/HFGI/editings/styleclip/global_directions/ffhq/fs3.npy',
		's_statistics': '/content/drive/MyDrive/HFGI_New/HFGI/editings/styleclip/global_directions/ffhq/S_mean_std',
		'templates': '/content/drive/MyDrive/HFGI_New/HFGI/editings/styleclip/global_directions/templates.txt'
	}
}