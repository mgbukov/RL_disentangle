LOGDIR="logs/STEFAN_direct_16q_3000iters_L_eta4.1_checkpoint6000_128envs_60steps_fvf"
OUTDIR="data/results"

python3 evaluate.py \
    --checkpoint $LOGDIR/checkpoint9000.pickle \
    --config $LOGDIR/config.yaml \
    --num_qubits 16 \
    --output "$OUTDIR/agent-16q.json" \
    --max_steps 500 \
    --epsi 1e-3 \
    --n_tests 100 \
    --haar_random \
    --weakly_entangled \
    --eta 4.1
