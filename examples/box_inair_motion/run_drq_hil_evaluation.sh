export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/sebastiano/voxel-serl/examples/box_inair_motion/drq_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_in_air_rotation \
    --wandb_project "dual_inairrot_voxnet_pretrained" \
    --exp_name="InAirRot_VoxNet_1500_box50" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/sebastiano/voxel-serl/examples/box_inair_motion/VoxNet_pretrained/checkpoints dual_inairrot_voxnet_pretrained 0402-10:23"\
    --eval_checkpoint_step 1500 \
    --eval_n_trajs 30 \
    --evaluation \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
