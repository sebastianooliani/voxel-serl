export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_her_two_buffers.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot_motion_planning \
    --wandb_project "dual_sac_motion_planning" \
    --exp_name=sac_policy_motion_rrl \
    --max_traj_length 100 \
    --seed 42 \
    --random_steps 0 \
    --training_starts 200 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 100000 \
    --reward_scale 1 \
    --demo_paths /home/sebastiano/voxel-serl/examples/box_motion_her/dual_20_her_transitions_2025-04-18_15-46-42.pkl \
    # --debug
