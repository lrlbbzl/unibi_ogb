CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbl-wikikg2 \
--model UniBi_2 --score_rel False --rank 200 --learning_rate 1e-1 \
--batch_size 300 --optimizer Adagrad --regularizer DURA_UniBi_2 --lmbda 1e-2 \
--w_rel 0.25 --valid 1 
