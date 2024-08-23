export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name="SAC no images Evaluation" \
    --camera_mode none \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/experiment_setup/SAC/checkpoints SAC no images 0820-17:01"\
    --eval_checkpoint_step 40000 \
    --eval_n_trajs 30 \
    \
    --encoder_type none \
    --state_mask all \
#    --debug
