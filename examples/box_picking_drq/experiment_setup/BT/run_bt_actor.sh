export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bt_policy.py "$@" \
    --env box_picking_camera_env_dual_robot \
    --exp_name=bt_drq_policy \
    --max_traj_length 1000 \
    --eval_n_trajs 30 \
