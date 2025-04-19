export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_her_two_buffers.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --wandb_project "dual_sac_motion_planning" \
    --exp_name=sac_policy_motion_rrl_box52_30goal_eval_45000_unseen_adapt \
    --eval_checkpoint_path /home/sebastiano/voxel-serl/examples/box_motion_her/checkpoints_0418-15:53 \
    --eval_checkpoint_step 45000 \
    --eval_n_trajs 30 \
    --evaluation \
    --number_eval_points 30 \
    # --debug
