export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python /home/sebastiano/voxel-serl/examples/box_motion_her/drq_policy_her_two_buffers.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --wandb_project "dual_motion_voxnet_pretrained" \
    --exp_name="dual_motion_voxnet_pretrained_resl" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 500 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_motion_her/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_motion_her/dual_20_pcd_her_transitions_2025-04-09_09-04-28.pkl \
    --dual \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
