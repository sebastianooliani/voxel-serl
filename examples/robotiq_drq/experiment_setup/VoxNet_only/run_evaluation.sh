export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --actor \
    --env robotiq_camera_env \
    --exp_name="Voxnet only Evaluation" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/nico/real-world-rl/serl/examples/robotiq_drq/experiment_setup/VoxNet_only/checkpoints voxnet only 0822-13:43"\
    --eval_checkpoint_step 14000 \
    --eval_n_trajs 30 \
    \
    --encoder_type voxnet \
    --state_mask none \
    --encoder_bottleneck_dim 128 \
#    --debug