export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bt_policy.py "$@" \
    --dual \
    --task inairrot \
    --env box_picking_camera_env_dual_robot_in_air_rotation \
    --wandb_project bt-inairrot \
    --exp_name=bt_inairrot_policy_box53_adapt \
    --max_traj_length 100 \
    --eval_n_trajs 30 \
    # --debug
