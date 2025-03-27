export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot \
    --wandb_project "dual_sac_lift" \
    --exp_name=sac_policy_lift \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 10000 \
    --random_steps 0 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --eval_period 1000 \
    --reward_scale 1 \
    # --debug