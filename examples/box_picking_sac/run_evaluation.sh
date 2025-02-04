export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
python sac_policy_hil.py "$@" \
    --actor \
    --env box_picking_camera_env_dual_robot_reorientation \
    --wandb_project "dual_sac_reorientation" \
    --exp_name=sac_policy_reorient_eval_25000 \
    --eval_checkpoint_path /home/sebastiano/voxel-serl/examples/box_picking_sac/checkpoints_0204-09:58\
    --eval_checkpoint_step 25000 \
    --eval_n_trajs 10 \
    #--debug
