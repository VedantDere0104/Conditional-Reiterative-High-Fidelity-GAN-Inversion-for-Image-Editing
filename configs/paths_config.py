dataset_paths = {
	#  Face Datasets (FFHQ - train, CelebA-HQ - test)
	'ffhq': 'ffhq_dataset',
	'ffhq_val': 'celebhq_dataset',

	#  Cars Dataset (Stanford cars)
	'cars_train': 'cars_dataset',
	'cars_val': 'cars_test',
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
		'delta_i_c': './editings/styleclip/global_directions/ffhq/fs3.npy',
		's_statistics': './editings/styleclip/global_directions/ffhq/S_mean_std',
		'templates': './editings/styleclip/global_directions/templates.txt'
	}
}
