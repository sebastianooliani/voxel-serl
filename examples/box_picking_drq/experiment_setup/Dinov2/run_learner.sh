export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python /home/sebastiano/voxel-serl/examples/box_picking_drq/drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="dinov2_one_arm" \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 96 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/Dinov2/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_picking_drq/box_picking_20_demos_2024-11-20_14-50-05_onearm_rgb_128.pkl \
    \
    --encoder_type dinov2 \
    --encoder_bottleneck_dim 128 \
    --debug
