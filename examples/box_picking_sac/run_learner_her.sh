export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_her.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --exp_name=sac_drq_policy_her \
    --max_traj_length 300 \
    --seed 42 \
    --training_starts 900 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths /home/sebastiano/voxel-serl/examples/box_picking_bc/ur5_test_20_demos_2024-12-11_16-23-29_her.pkl \
    --debug