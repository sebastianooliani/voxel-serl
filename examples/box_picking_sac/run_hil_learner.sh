export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_hil.py "$@" \
    --learner \
    --env box_picking_camera_env_dual_robot \
    --wandb_project "dual_sac_reorientation" \
    --exp_name=sac_policy_reorient \
    --max_traj_length 300 \
    --seed 42 \
    --training_starts 900 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths /home/sebastiano/voxel-serl/examples/box_picking_bc/ur5_test_2_demos_2025-01-29_09-06-34_hil_test.pkl \
    --debug
