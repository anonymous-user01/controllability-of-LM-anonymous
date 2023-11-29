'''
- CUDA_VISIBLE_DEVICES=5
- my_seed=47
- model=gpt2_small_init_weight
- dataset=topic-1
- rl_batch_size=256
- rl_lr=5e-04
- rl_num_epoch=20
- gen_len=15
- dropout=quantile
- dropout_rate={0.95, 0.9, 0.85, 0.8, 0.0}
- init_weight={uniform, normal, Glorot, He, constant}
'''
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=uniform --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.95
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=normal --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.95
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=Glorot --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.95
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=He --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.95
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=constant --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.95

CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=uniform --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.9
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=normal --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.9
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=Glorot --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.9
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=He --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.9
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=constant --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.9

CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=uniform --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.85
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=normal --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.85
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=Glorot --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.85
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=He --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.85
# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=constant --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.85

# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=uniform --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.8
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=normal --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.8
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=Glorot --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.8
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=He --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.8
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=constant --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.8

# CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=uniform --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.0
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=normal --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.0
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=Glorot --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.0
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=He --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.0
# # CUDA_VISIBLE_DEVICES=5 taskset -c 1-128 python3 main_LLM.py --my_seed=47 --task=rl --model=gpt2_small_init_weight --init_weight=constant --dataset=topic-1 --batch_size=128 --lr=1e-06 --num_epoch=20 --num_patience=3 --rl_batch_size=256 --rl_lr=5e-04 --rl_num_epoch=20 --decoding=stochastic --prefix_len=2 --gen_len=15 --dropout=quantile --dropout_rate=0.0