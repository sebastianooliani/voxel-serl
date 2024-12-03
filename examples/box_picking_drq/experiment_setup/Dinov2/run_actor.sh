export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python /home/sebastiano/voxel-serl/examples/box_picking_drq/drq_policy.py "$@" \
    --actor \
    --env box_picking_camera_env \
    --exp_name="dinov2_one_arm" \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 20000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 96 \
    --eval_period 0 \
    \
    --encoder_type dinov2 \
    --encoder_bottleneck_dim 128 \
    --debug
