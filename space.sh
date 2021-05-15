CUDA_VISIBLE_DEVICES=0 python experiment/run_baseopt.py -m  MASQL_MASQL_MASQL  -s2 2.0 -ms 5000 >/dev/null & 
CUDA_VISIBLE_DEVICES=1 python experiment/run_baseopt.py -m MADDPG_MADDPG_MADDPG  -s2 2.0 -ms 5000 >/dev/null &
CUDA_VISIBLE_DEVICES=2 python experiment/run_baseopt.py -m PR2AC0_PR2AC0_PR2AC0  -s2 2.0 -ms 5000 >/dev/null &
CUDA_VISIBLE_DEVICES=3 python experiment/run_baseopt.py -m DDPG_DDPG_DDPG  -s2 2.0 -ms 5000  >/dev/null &
# CUDA_VISIBLE_DEVICES=0 python experiment/run_optuna.py -m ROMMEO_ROMMEO_ROMMEO  -s2 2.0 -ms 5000 >/dev/null &


# CUDA_VISIBLE_DEVICES=1 python experiment/run_Jbaseopt.py -m JSQL -do_nego 0 -s2 1.5 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=3 python experiment/run_Jbaseopt.py -m JSQL -do_nego 0 -s2 1 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=1 python experiment/run_Jbaseopt.py -m JSQL -do_nego 0 -s2 2 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=1 python experiment/run_Jbaseopt.py -m GPF -do_nego 0 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=2 python experiment/run_Jbaseopt.py -m GPF -do_nego 1 -ms 5000 >/dev/null &

# CUDA_VISIBLE_DEVICES=2 python experiment/run_ss_baseopt.py -m JSQL -s2 1.5 -ms 5000 >/dev/null &

# CUDA_VISIBLE_DEVICES=0 python experiment/run_ss_baseopt.py -m GPF -s2 1.5 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=1 python experiment/run_ss_baseopt.py -m GPF -s2 1 -ms 5000 >/dev/null &
# CUDA_VISIBLE_DEVICES=2 python experiment/run_ss_baseopt.py -m GPF -s2 2 -ms 5000 >/dev/null &