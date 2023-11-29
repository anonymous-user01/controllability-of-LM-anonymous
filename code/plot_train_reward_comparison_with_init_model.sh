'''
- plot_model={gpt2_small, opt, xglm, gpt2_large, gpt2_small_init_weight=uniform}
- dataset={topic-1, topic-3}
- history={reward, acc}
'''
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small_init_weight=uniform --dataset=topic-1 --history=reward
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small_init_weight=uniform --dataset=topic-1 --history=acc
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small --dataset=topic-1 --history=reward
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small --dataset=topic-1 --history=acc

python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small_init_weight=uniform --dataset=topic-3 --history=reward
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small_init_weight=uniform --dataset=topic-3 --history=acc
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small --dataset=topic-3 --history=reward
python3 plot_train_reward_comparison_by_init_model.py --my_seed=47 --plot_model=gpt2_small --dataset=topic-3 --history=acc