# Regularized Opponent Model with Maximum Entropy Objective

This repo aims to provide an algorithm implementation for IJCAI 2019 paper
[Regularized Opponent Model with Maximum Entropy Objective (ROMMEO)](https://arxiv.org/abs/1905.08087) and its baselines.

There are some additional materials avaiable here:

* [Paper & Appendix](./ROMMEO_Appendix.pdf)
* [Slides](./ROMMEO_Slides.pdf)
* [Poster](./ROMMEO_Poster.pdf)

## Installation

1. Clone rllrb
  
 ```shell
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
sudo pip3 install -e .
 ```

 2. Intsall other dependencies
   
 ```shell
sudo pip3 install joblib path.py gtimer theano keras tensorflow gym  tensorflow_probability
 ```

 3. Intsall maci
   
 ```shell
cd rommeo
pip3 install mujoco-py
sudo pip3 install -e .
 ```


## Runing Experiments

```shell
cd experiment
python3 run_rommeo.py
python3 run_baseline.py
```
    
