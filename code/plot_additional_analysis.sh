# pos_game_data.py (reward_order = 1, 2, 4, 16 / min_reward_action = 1)
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=1
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=2
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=4
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_order=16

# train_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=2
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=4
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=16

# evalute_pos_game_agent.py (reward_order = 1, 2, 4, 16 / case = 5 / min_reward_action = 1)
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=1
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=2
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=4
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_order=16

# plot_additional_analysis.py
CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=1
