export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot \
    --exp_name=sac_drq_policy_evaluation \
    --eval_checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_sac/checkpoints_1108-14:55\
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 10 \
    #--debug
