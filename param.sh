# CUDA_VISIBLE_DEVICES=0 python experiment/run_optuna.py &
CUDA_VISIBLE_DEVICES=0 python experiment/run_baseopt.py -m PR2AC0_PR2AC0 &
CUDA_VISIBLE_DEVICES=1 python experiment/run_baseopt.py -m MASQL_MASQL & 
CUDA_VISIBLE_DEVICES=2 python experiment/run_baseopt.py -m MADDPG_MADDPG & 
CUDA_VISIBLE_DEVICES=3 python experiment/run_baseopt.py -m DDPG_DDPG & 
CUDA_VISIBLE_DEVICES=3 python experiment/run_baseopt.py -m DDPG-OM_DDPG-OM & 