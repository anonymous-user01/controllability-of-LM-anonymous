# Reward 함수가 직접 짠 Exp 함수인 경우

# # pos_game_data.py (reward_order = 1, 2, 4, 16 / min_reward_action = 1)
# # python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=1
# # python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=6
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=4
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=8
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=12
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=16

# # train_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# # CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# # CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=4
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=8
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=12
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=16

# # evalute_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# # CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# # CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=4
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=8
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=12
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=16

# # plot_additional_analysis.py
# CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=1

###############################################################################################################################################################################################################
# Reward 함수가 Beta 함수인 경우
# Row 1 in Figure 3

# # pos_game_data.py (reward_order = 1, 2, 4, 16 / min_reward_action = 1)
# # python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=1
# # python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=6
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.8
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.6
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.4
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.2

# # train_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# # CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# # CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.8
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.6
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.4
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.2

# # evalute_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# # CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# # CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.8
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.6
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.4
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.25 --reward_beta=0.2

# # plot_additional_analysis.py
# CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=1

# Row 2 in Figure 3
# pos_game_data.py (reward_order = 1, 2, 4, 16 / min_reward_action = 1)
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=1
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=6
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# train_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# evalute_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=6
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# plot_additional_analysis.py
CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=1

###############################################################################################################################################################################################################

# # pos_game_data.py (reward_order = 1, 2, 4, 16 / min_reward_action = 1)
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=1
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=3 --reward_order=1
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=5 --reward_order=1
# python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=7 --reward_order=1

# # train_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=3 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=5 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=7 --reward_order=1

# # evalute_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=3 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=5 --reward_order=1
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=7 --reward_order=1

# # plot_additional_analysis.py
# CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --reward_order=1
