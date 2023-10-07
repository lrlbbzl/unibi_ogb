CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbl-biokg \
--model UniBi_2 --score_rel False --rank 3000 --learning_rate 1e-1 \
--batch_size 500 --optimizer Adagrad --regularizer DURA_UniBi_2 --lmbda 5e-3 \
--w_rel 0.25 --valid 1 
