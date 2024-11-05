export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/sebastiano/voxel-serl/examples/box_picking_drq/drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="voxnet pretrained" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_picking_drq/box_picking_20_demos_2024-11-05_13-27-29.pkl \
    \
    --encoder_type voxnet-pretrained \
    --state_mask all \
    --encoder_bottleneck_dim 128 \
    --debug
