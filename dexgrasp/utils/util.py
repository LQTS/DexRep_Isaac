import numpy as np
import math
import torch

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def split_batch_process(split_batch, input_tensor_list, process_chain):
    batch_num = input_tensor_list[0].shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    split_input_list = []
    for it in input_tensor_list:
        assert batch_num == it.shape[0]
        it_slices = [
            it[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
        ]
        split_input_list.append(it_slices)
    # transpose input list
    split_input_list = [list(item) for item in zip(*split_input_list)]
    # process
    result_list = []
    for iter in range(batch_split_num):
        chain_idx = 0
        result = None
        for my_func in process_chain:
            if chain_idx == 0: # input
                result = my_func(*split_input_list[iter])
            else:
                result = my_func(result)
            chain_idx += 1
        result_list.append(result)

def split_torch_dist(finger_points, obj_pcb, split_batch):
    assert finger_points.shape[0] == obj_pcb.shape[0]
    batch_num = finger_points.shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    finger_points_split = [
        finger_points[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
    ]
    obj_pcb_split = [
        obj_pcb[i * split_batch:(i + 1) * split_batch, ...] \
            for i in range(batch_split_num)
    ]
    # process
    dis_min_slices, dis_min_idx_slices = [], []
    for iter in range(batch_split_num):
        dis_split = torch.cdist(finger_points_split[iter], obj_pcb_split[iter])
        dis_min_split, dis_min_idx_split = torch.min(dis_split, dim=-1)
        dis_min_slices.append(dis_min_split)
        dis_min_idx_slices.append(dis_min_idx_split)
    # cat
    dis_min = torch.cat(dis_min_slices, dim=0)
    dis_min_idx = torch.cat(dis_min_idx_slices, dim=0)

    return dis_min, dis_min_idx