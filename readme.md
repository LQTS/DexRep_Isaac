# Dependencies
- Create a conda environment 
    ```shell
    conda create -n your_env_name python==3.8
    conda activate your_env_name
    ```
- Install torch
    ```shell
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
- Install IsaacGym

    1. Download [isaacgym](https://developer.nvidia.com/isaac-gym/download) 
    2. Extract the downloaded files to the main directory of the project
    3. Use the following commands to install isaacgym  
  ```shell
    cd isaacgym/python
    pip install -e .
    ```
- Install DexRep
    ```shell
    cd dexgrasp
    pip install -e .
    ```
The above commands show how to install the major packages. You can install other packages by yourself if needed.

# Run the scripts
```
cd dexgrasp
python train.py --task=ShadowHandGraspDexRep --algo=ppo1 --seed=0 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_20
```
Notes:
- If you do not want to open the simulator windows, add **--headless** 
- More parameters can be found in **dexgrasp/cfg/shadow_hand_grasp_dexrep.yaml**