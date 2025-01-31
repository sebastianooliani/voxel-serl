export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/sebastiano/voxel-serl/examples/box_picking_sac/sac_policy_images.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot \
    --wandb_project "dual_lift_voxnet_pretrained" \
    --exp_name="sac dual lift voxnet pretrained" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 50000 \
    --reward_scale 1 \
    --random_steps 0 \
    --training_starts 900 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --checkpoint_period 500 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_picking_drq/box_picking_20_demos_2025-01-24_16-00-04_dual_lift_pcd.pkl \
    --dual \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    --debug
