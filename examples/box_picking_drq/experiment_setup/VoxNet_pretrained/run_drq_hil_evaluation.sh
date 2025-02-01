export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python /home/sebastiano/voxel-serl/examples/box_picking_drq/drq_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot \
    --exp_name="Voxnet Pretrained Evaluation 2500" \
    --wandb_project "dual_lift_voxnet_pretrained" \
    --camera_mode pointcloud \
    --batch_size 128 \
    --max_traj_length 100 \
    --checkpoint_path "/home/sebastiano/voxel-serl/examples/box_picking_drq/experiment_setup/VoxNet_pretrained/checkpoints dual lift voxnet pretrained 0201-09:02"\
    --eval_checkpoint_step 2500 \
    --eval_n_trajs 10 \
    --evaluation\
    --encoder_type voxnet-pretrained \
    --state_mask dual \
    --encoder_bottleneck_dim 128 \
#    --debug
