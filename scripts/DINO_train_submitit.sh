python run_with_submitit.py --timeout 3000 --job_name DINO \
	--job_dir /home/pmh/code/DINO_Grasp/log --ngpus 2 --nodes 1 \
	-c config/DINO/DINO_4scale.py --data_path /home/pmh/data/soft_grasp_data \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
