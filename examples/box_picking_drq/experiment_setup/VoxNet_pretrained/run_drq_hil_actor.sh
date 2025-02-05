export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python /home/sebastiano/voxel-serl/examples/box_picking_drq/drq_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_reorientation \
    --wandb_project "dual_reorient_voxnet_pretrained" \
    --exp_name="dual reorient voxnet pretrained" \
    --camera_mode pointcloud \
    --max_traj_length 100 \
    --seed 42 \
    --max_steps 25000 \
    --random_steps 0 \
    --training_starts 500 \
    --utd_ratio 8 \
    --batch_size 128 \
    --eval_period 0 \
    --dual \
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
    # --debug
