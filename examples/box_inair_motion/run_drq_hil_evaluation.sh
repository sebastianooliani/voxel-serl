export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python /home/sebastiano/voxel-serl/examples/box_inair_motion/drq_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_in_air_rotation \
    --wandb_project "dual_inairrot_voxnet_pretrained" \
    --exp_name="InAirRot_VoxNet_3500_box54_adapt" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/sebastiano/voxel-serl/examples/box_inair_motion/VoxNet_pretrained/checkpoints dual_inairrot_voxnet_pretrained 0408-08:36"\
    --eval_checkpoint_step 3500 \
    --eval_n_trajs 30 \
    --evaluation \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
