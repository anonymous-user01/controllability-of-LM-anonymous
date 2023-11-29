'''
- dataset={topic-1, topic-3}
- history={reward, acc}
'''
python3 plot_train_reward_comparison_by_model_all.py --my_seed=47 --dataset=topic-1 --history=reward
python3 plot_train_reward_comparison_by_model_all.py --my_seed=47 --dataset=topic-1 --history=acc
python3 plot_train_reward_comparison_by_model_all.py --my_seed=47 --dataset=topic-3 --history=reward
python3 plot_train_reward_comparison_by_model_all.py --my_seed=47 --dataset=topic-3 --history=acc