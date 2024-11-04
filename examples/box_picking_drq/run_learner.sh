export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python drq_policy.py "$@" \
    --learner \
    --env box_picking_camera_env \
    --exp_name="ResNet18 feat red 32 128" \
    --camera_mode rgb \
    --max_traj_length 100 \
    --seed 1 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 96 \
    --checkpoint_period 1000 \
    --checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/ResNet18/checkpoints \
    --demo_path /home/sebastiano/voxel-serl/examples/box_picking_drq/box_picking_20_demos_2024-11-04_11-38-50.pkl \
    \
    --encoder_type resnet-pretrained-18 \
    --encoder_bottleneck_dim 128 \
    --state_mask all \
    --encoder_kwargs pooling_method \
    --encoder_kwargs feature_reduction \
    --encoder_kwargs num_kp \
    --encoder_kwargs 32 \
    --debug