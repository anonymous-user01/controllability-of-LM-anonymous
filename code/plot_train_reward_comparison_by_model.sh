'''
- plot_model={gpt2_small, opt, xglm, gpt2_large}
- dataset={topic-1, topic-3}
- history={reward, acc}
'''
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=gpt2_large --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=gpt2_large --history=acc
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=gpt2_large --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=gpt2_large --history=acc

python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=gpt2_small --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=gpt2_small --history=acc
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=gpt2_small --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=gpt2_small --history=acc

python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=opt --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=opt --history=acc
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=opt --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=opt --history=acc

python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=xglm --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=xglm --history=acc
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=xglm --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=xglm --history=acc

python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=all --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-1 --plot_model=all --history=acc
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=all --history=reward
python3 plot_train_reward_comparison_by_model.py --my_seed=47 --dataset=topic-3 --plot_model=all --history=acc
