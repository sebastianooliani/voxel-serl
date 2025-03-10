export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bt_policy.py "$@" \
    --dual \
    --env box_picking_camera_env_dual_robot_reorientation \
    --task reorient \
    --wandb_project bt \
    --exp_name=bt_drq_policy \
    --max_traj_length 100 \
    --eval_n_trajs 20 \
    --debug
