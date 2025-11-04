python3 run_train.py --USE_WANDB=True \
                    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
                    --blocks=16 --bsz=32 --d_model=512 --dataset=lobster-prediction --merging=padded \
                    --dir_name='/home/myuser/processed_data/AMZN/2024_Dec' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True  \
                    --masking=none \
                    --num_devices=8 --n_data_workers=8 \
                    --debug_loading=False \
                    --enable_profiler=False \
                    --random_offsets_train=True \
                    --shuffle_train=True \
                    --debug_overfit=False \
                    --restore='/home/myuser/checkpoints/lyric-pyramid-35_vqddc48h/' \
                    --restore_step=25 \
                    # --curtail_epochs=0 \
                    # --restore='checkpoints/eager-shadow-750_af39bb9u/'


