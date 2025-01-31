export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/sebastiano/voxel-serl/examples/box_picking_sac/sac_policy_images.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot \
    --wandb_project "dual_lift_voxnet_pretrained" \
    --exp_name="sac dual lift voxnet pretrained" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 10000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --reward_scale 1 \
    --eval_period 1000 \
    --dual \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    --debug
