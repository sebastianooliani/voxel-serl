export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python bc_policy.py "$@" \
    --env box_picking_camera_env_dual_robot \
    --exp_name=bc_drq_policy \
    --seed 42 \
    --batch_size 256 \
    --demo_paths "ur5_test_20_demos_2024-11-08_14-17-36.pkl" \
    --eval_checkpoint_step 0 \
    #--debug # wandb is disabled when debug
