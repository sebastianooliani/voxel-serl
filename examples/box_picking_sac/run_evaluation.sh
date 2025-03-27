export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot\
    --wandb_project "dual_sac_lift" \
    --exp_name=sac_policy_lift_eval_10000_box_50_seen \
    --eval_checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_sac/checkpoints_0311-10:38 \
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 30 \
    # --debug
