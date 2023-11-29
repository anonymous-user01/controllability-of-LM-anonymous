'''
- CUDA_VISIBLE_DEVICES=5
- model=xglm
- dataset=topic-1
- rl_batch_size=128
- rl_lr=5e-05
- rl_num_epoch=5
'''
CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=xglm --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=128 --rl_lr=5e-05 --rl_num_epoch=5 --decoding=stochastic --prefix_len=2 --gen_len=13 --dropout=quantile --dropout_rate=0.0
CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=xglm --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=128 --rl_lr=5e-05 --rl_num_epoch=5 --decoding=stochastic --prefix_len=2 --gen_len=13 --dropout=quantile --dropout_rate=0.8
CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=xglm --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=128 --rl_lr=5e-05 --rl_num_epoch=5 --decoding=stochastic --prefix_len=2 --gen_len=13 --dropout=quantile --dropout_rate=0.85
CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=xglm --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=128 --rl_lr=5e-05 --rl_num_epoch=5 --decoding=stochastic --prefix_len=2 --gen_len=13 --dropout=quantile --dropout_rate=0.9
CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=xglm --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=128 --rl_lr=5e-05 --rl_num_epoch=5 --decoding=stochastic --prefix_len=2 --gen_len=13 --dropout=quantile --dropout_rate=0.95
