export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_her.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --wandb_project "dual_sac_motion_planning" \
    --exp_name=sac_policy_motion_box50_eval_35000_seen \
    --eval_checkpoint_path /home/sebastiano/voxel-serl/examples/box_motion_her/checkpoints_0329-10:17 \
    --eval_checkpoint_step 35000 \
    --eval_n_trajs 30 \
    --evaluation \
    --number_eval_points 30 \
    # --debug
