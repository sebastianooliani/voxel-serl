export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python drq_policy_her.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --wandb_project "dual motion voxnet pretrained" \
    --exp_name=sac_policy_motion_box50_eval_2000_seen \
    --camera_mode pointcloud \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_motion_her/checkpoints_dual_motion_voxnet_pretrained_0326-14:16 \
    --eval_checkpoint_step 2000 \
    --eval_n_trajs 30 \
    --evaluation \
    --dual \
    --number_eval_points 30 \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
