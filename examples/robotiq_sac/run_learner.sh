export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_robotiq.py "$@" \
    --learner \
    --env robotiq-grip-v1 \
    --exp_name=sac_husky \
    --max_traj_length 100 \
    --seed 1 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths "/home/tuvok/build_playground/real-world-rl/serl/examples/robotiq_sac/robotiq_test_20_demos_2024-06-13_16-30-05.pkl" \
    --checkpoint_path "/home/tuvok/build_playground/real-world-rl/serl/examples/robotiq_sac/checkpoints" \
    --log_rlds_path "/home/tuvok/build_playground/real-world-rl/serl/examples/robotiq_sac/rlds"
#    --debug
