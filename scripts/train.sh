cd ..
# python3 run.py \
#     --seed 7 --num_qubits 6 --num_envs 512 --steps 90 --steps_limit 90 --num_iters 12000 \
#     --attn_heads 4 --transformer_layers 4 --embed_dim 256 --dim_mlp 1024 \
#     --batch_size 512 --pi_lr 2e-4 --entropy_reg 0.1 --obs_fn rdm_2q_mean_real \
#     --state_generator haar_unif --min_entangled 3 --p_gen 0.3 \
#     --log_every 100 --checkpoint_every 100 \
#     --suffix sample --demo_every 100

# python3 run.py \
#     --seed 4 --num_qubits 5 --num_envs 128 --steps 64 --steps_limit 40 --num_iters 10000 \
#     --attn_heads 4 --transformer_layers 4 --embed_dim 128 --dim_mlp 512 \
#     --batch_size 128 --pi_lr 2e-4 --entropy_reg 0.1 --obs_fn rdm_2q_mean_real \
#     --state_generator haar_unif --min_entangled 2 --p_gen 0.3 \
#     --log_every 100 --demo_every 100 --suffix sample

python3 run.py \
    --seed 0 --num_qubits 4 --num_envs 128 --steps 8 --steps_limit 16 --num_iters 4000 \
    --state_generator haar_unif --obs_fn rdm_2q_rsqr_nisq_mean_real \
    --attn_heads 2 --transformer_layers 2 --embed_dim 128 --dim_mlp 256 \
    --batch_size 128 --pi_lr 2e-4 --entropy_reg 0.1  --suffix sample \
    --log_every 100 --demo_every 100