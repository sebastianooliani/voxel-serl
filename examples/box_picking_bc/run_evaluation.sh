export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python bc_policy.py "$@" \
    --env box_picking_camera_env \
    --exp_name=bc_drq_dual_policy \
    --checkpoint_path "/home/sebastiano/voxel-serl/examples/box_picking_bc/"\
    --eval_checkpoint_step 90000 \
    --eval_n_trajs 10 \
    --debug
