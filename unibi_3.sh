CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbl-biokg \
--model UniBi_3 --score_rel False --rank 4000 --learning_rate 1e-2 --max_epochs 60 \
--batch_size 500 --optimizer Adagrad --regularizer DURA_UniBi_3 --lmbda 5e-3 \
--w_rel 0.25 --valid 1 --rel_norm --seed 0
