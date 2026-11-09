set LOGDIR=logs\checkpointed_12q_10000iters_XL_eta4.1_v2.1_tfvf_continue
set OUTDIR=logs\checkpointed_12q_10000iters_XL_eta4.1_v2.1_tfvf_continue


python evaluate.py ^
    --config "%LOGDIR%\config.yaml" ^
    --checkpoint "%LOGDIR%\checkpoint13500.pickle" ^
    --num_qubits 12 ^
    --output "%OUTDIR%\results-checkpoint13500.json" ^
    --max_steps 250 ^
    --epsi 1e-3 ^
    --n_tests 200 ^
    --haar_random ^
    --weakly_entangled ^
    --eta 4.2 4.1 4.0 ^
    --subsystem_size 2 3 4 ^
    --cuda
