export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy_hil.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot_in_air_rotation \
    --wandb_project "dual_inairrot_voxnet_pretrained" \
    --exp_name="dual_inairrot_voxnet_pretrained" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 500 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_picking_drq/box_picking_20_demos_2025-02-28_11-03-06_dual_lift_pcd.pkl \
    --dual \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
