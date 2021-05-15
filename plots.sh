# CUDA_VISIBLE_DEVICES=0 python experiment/run_optuna.py &
# CUDA_VISIBLE_DEVICES=1 python -m runx.runx ymls/maddpg_30.yml &
# CUDA_VISIBLE_DEVICES=1 python -m runx.runx ymls/pr2_30.yml &
# CUDA_VISIBLE_DEVICES=0 python -m runx.runx ymls/gpf_30.yml &
# CUDA_VISIBLE_DEVICES=3 python -m runx.runx ymls/jsql_30.yml &
# CUDA_VISIBLE_DEVICES=3 python -m runx.runx ymls/rommeo_30.yml &
# CUDA_VISIBLE_DEVICES=0 python -m runx.runx ymls/masql_30.yml &

CUDA_VISIBLE_DEVICES=1 python -m runx.runx ymls/accf_1_30.yml &
CUDA_VISIBLE_DEVICES=2 python -m runx.runx ymls/accf_1_20.yml &
