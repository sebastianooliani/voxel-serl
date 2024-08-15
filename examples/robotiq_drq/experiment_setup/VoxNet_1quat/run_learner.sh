export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python /home/nico/real-world-rl/serl/examples/robotiq_drq/drq_policy_robotiq.py "$@" \
    --learner \
    --env robotiq_camera_env \
    --exp_name="voxnet only pure 32 16 8" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 24 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --checkpoint_period 2500 \
    --demo_path /home/nico/real-world-rl/serl/examples/robotiq_drq/pcb_insert_20_demos_aug13_all_1quad.pkl \
    \
    --encoder_type voxnet \
    --state_mask none \
    --encoder_bottleneck_dim 128 \
    --proprio_latent_dim 0 \
    --enable_obs_rotation_wrapper True \
    --enable_obs_rotation_augmentation False \
#    --debug