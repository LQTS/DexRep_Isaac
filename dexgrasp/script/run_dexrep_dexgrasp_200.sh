python train.py \
--task=ShadowHandGraspDexRepDexgrasp \
--algo=ppo1 \
--seed=0 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--logdir=logs/dexrep_dexgrasp_500_update \
--num_objs=500 \
--headless

#cd /remote-home/liuqingtao/UniDexGrasp/dexgrasp_policy/dexgrasp
#conda activate dexgrasp
#python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=42 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp_500_update_seed42 --num_objs=500 --headless
#python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=111 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp_20_update_seed111 --num_objs=20 --headless
#python train.py --task=ShadowHandGraspDexRepDexgrasp --algo=ppo1 --seed=222 --rl_device=cuda:0 --sim_device=cuda:0 --logdir=logs/dexrep_dexgrasp_20_update_seed222 --num_objs=20 --headless
