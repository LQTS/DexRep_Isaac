import json
import os
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
import torch
from typing import Callable, List, Tuple
# from model.utils.extraction import instantiate_extractor
# from model.reproductions.vmvp import VMVP
# from model.vitac.vtt_repic import VTT_RePic
# from model.vitac.vtt_reall import VTT_ReAll
# from model.vitac.t_retac import T_ReTac

import torch.nn.functional as F
import torchvision.transforms as T
# from torchvision.models import get_model, list_models, ResNet
import torchvision.models as models

from einops import rearrange
import numpy as np
DEFAULT_CACHE = "cache/"
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#

class Encoder_DexRep(nn.Module):
    """return features of provided states and point clouds of current object and goal"""

    def __init__(self, state_dim, args):
        super(Encoder_DexRep, self).__init__()
        self.args = args
        # two-stream pointnet
        # self.pointnet = PointNetfeatTwoStream(
        #     output_dim=args.pointnet_output_dim)  # expects [batch, 3, num_points]
        # projection layers
        self.dexrep_sensor = nn.Linear(1080, args.pointnet_output_dim // 2)
        self.dexrep_pointL = nn.Linear(1280, args.pointnet_output_dim // 2)
        self.states_fc = nn.Linear(state_dim, args.pointnet_output_dim)

    def forward(self, x):
        # slice inputs into different parts
        obs_dict = self.args.flat2dict(x)
        states, goal = obs_dict['minimal_obs'], obs_dict['desired_goal']

        sensor, pointL = obs_dict['dexrep'][..., :1080].clone(), obs_dict['dexrep'][..., 1080:].clone()

        sensor_feat = F.relu(self.dexrep_sensor(sensor))
        sensor_feat = F.normalize(sensor_feat, dim=-1)

        pointL_feat = F.relu(self.dexrep_pointL(pointL))
        pointL_feat = F.normalize(pointL_feat, dim=-1)

        # get states features ========================================================
        # get states feature
        states_feat = F.relu(self.states_fc(torch.cat([states, goal], dim=-1)))
        states_feat = F.normalize(states_feat, dim=-1)

        return torch.cat([states_feat, sensor_feat, pointL_feat], dim=-1)


class PointNetfeat(nn.Module):
    def __init__(self, input_dim=3, output_dim=1024):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = self.bn3(self.conv3(out2))
        return out3  # return non-aggregated per-point features


class PointNetfeatTwoStream(nn.Module):
    """return a global feature vector describing the two input point clouds (same with point-to-point correspondence but rotated with a unknown matrix)"""

    def __init__(self, output_dim=1024, **kwargs):
        super(PointNetfeatTwoStream, self).__init__()
        self.output_dim = output_dim
        # pointnet model for extracting local (per-point) features
        self.feat_net = PointNetfeat(input_dim=12, output_dim=output_dim)

    def aggregate(self, x):
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, self.output_dim)

    def forward(self, point_set1, point_set2):
        """shape of input: [num_samples, 3, num_points]"""
        local_features = self.feat_net(
            torch.cat([point_set1, point_set2], dim=1))
        global_features = self.aggregate(local_features)
        return global_features

class Encoder_GeoDex_cold(nn.Module):
    """return features of provided states and point clouds of current object and goal"""

    def __init__(self, env_cfg, encoder_cfg):
        super(Encoder_GeoDex_cold, self).__init__()
        self.num_points = env_cfg["geodex"]["sample_num_points"]
        # two-stream pointnet
        self.pointnet = PointNetfeatTwoStream(
            output_dim=encoder_cfg["pointnet_output_dim"])  # expects [batch, 3, num_points]
        # projection layers
        self.points_fc = nn.Linear(
            encoder_cfg["pointnet_output_dim"], encoder_cfg["obs_emb_dim"])

    @torch.no_grad()
    def forward_obs(self, obj_pnts, obj_norms, goal_pnts, goal_norms):
        batch_num = obj_pnts.shape[0]
        # reshape points
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(
            goal_norms.shape) == 2
        obj_normals = obj_norms.reshape(
            [batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape(
            [batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_points = torch.cat([obj_points, obj_normals], dim=-1)
        target_points = torch.cat(
            [target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_points = obj_points.transpose(2, 1)
        target_points = target_points.transpose(2, 1)
        points_feat = self.pointnet(obj_points, target_points)
        # get projected points feature
        points_emb = F.relu(self.points_fc(points_feat))
        points_emb = F.normalize(points_emb, dim=-1)
        return points_emb, points_feat

    def forward(self, feat): # need grad
        points_emb = F.relu(self.points_fc(feat))
        points_emb = F.normalize(points_emb, dim=-1)
        return points_emb

class Encoder_GeoDex(nn.Module):
    """return features of provided states and point clouds of current object and goal"""

    def __init__(self, env_cfg, encoder_cfg):
        super(Encoder_GeoDex, self).__init__()
        self.num_points = env_cfg["geodex"]["sample_num_points"]
        # two-stream pointnet
        self.pointnet = PointNetfeatTwoStream(
            output_dim=encoder_cfg["pointnet_output_dim"])  # expects [batch, 3, num_points]
        # projection layers
        self.points_fc = nn.Linear(
            encoder_cfg["pointnet_output_dim"], encoder_cfg["obs_emb_dim"])

    @torch.no_grad()
    def forward_obs(self, obj_pnts, obj_norms, goal_pnts, goal_norms):
        batch_num = obj_pnts.shape[0]
        # reshape points
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(
            goal_norms.shape) == 2
        obj_normals = obj_norms.reshape(
            [batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape(
            [batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_points = torch.cat([obj_points, obj_normals], dim=-1)
        target_points = torch.cat(
            [target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_points = obj_points.transpose(2, 1)
        target_points = target_points.transpose(2, 1)

        with torch.no_grad():
            points_feat = self.pointnet(obj_points, target_points)
            # get projected points feature
            points_emb = F.relu(self.points_fc(points_feat))
            points_emb = F.normalize(points_emb, dim=-1)
        return points_emb

    def forward(self, obj_pnts, obj_norms, goal_pnts, goal_norms): # need grad
        batch_num = obj_pnts.shape[0]
        # reshape points
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(
            goal_norms.shape) == 2
        obj_normals = obj_norms.reshape(
            [batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape(
            [batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_points = torch.cat([obj_points, obj_normals], dim=-1)
        target_points = torch.cat(
            [target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_points = obj_points.transpose(2, 1)
        target_points = target_points.transpose(2, 1)

        points_feat = self.pointnet(obj_points, target_points)
        # get projected points feature
        points_emb = F.relu(self.points_fc(points_feat))
        points_emb = F.normalize(points_emb, dim=-1)
        return points_emb


class Encoder_T(nn.Module):
    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls', f_ex_mode='MAP'):
        super(Encoder_T, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone, _, gap_dim = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode == 'patch':
            self.projector = nn.Sequential(
                instantiate_extractor(self.backbone, n_latents=1)(),
                nn.Linear(gap_dim, emb_dim))
        else:
            self.projector = nn.Linear(gap_dim, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.freeze = freeze
        # self.using_mp = using_mp

    @torch.no_grad()
    def forward(self, x):

        feat = self.backbone.get_representations(x, mode=self.en_mode)
        # if self.using_mp:
        #     feat = feat.mean(dim=1)
        return self.projector(feat), feat

    def forward_feat(self, feat):
        return self.projector(feat)

class Encoder_no_pre(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(Encoder_no_pre, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        # assert self.model_name in list_models(), f"{self.model_name} is not included in {list_models()}"
        # self.backbone = get_model(model_name) # using no weights

        if self.model_name == "resnet18":
            self.backbone = models.resnet18()
        else:
            raise AssertionError('Error model names!')
        gap_dim = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # abandon the last two layers

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(gap_dim, emb_dim)
        self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        if isinstance(x, torch.Tensor):
            imag, tac = x[:, :-20], x[:, -20:]
            imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        elif isinstance(x, tuple):
            imag, tac = x
        else:
            raise AssertionError
        imag = self.preprocess(imag)
        img_feat = self.backbone(imag)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.projector_img(img_feat)
        tac_feat = self.projector_tac(tac)

        feat = torch.cat([img_feat, tac_feat], dim=-1)


        return feat

class Encoder_CNN(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(Encoder_CNN, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        assert self.model_name =='CNN', f"{self.model_name} is not CNN"
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(64*56*56, emb_dim)
        self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        if isinstance(x, torch.Tensor):
            imag, tac = x[:, :-20], x[:, -20:]
            imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        elif isinstance(x, tuple):
            imag, tac = x
        else:
            raise AssertionError
        imag = self.preprocess(imag)
        img_feat = self.relu(self.conv1(imag))
        img_feat = self.maxpool(img_feat)
        img_feat = self.relu(self.conv2(img_feat))
        img_feat = self.maxpool(img_feat)
        img_feat = img_feat.contiguous().view(img_feat.size(0), -1)

        img_feat = self.projector_img(img_feat)
        tac_feat = self.projector_tac(tac)

        feat = torch.cat([img_feat, tac_feat], dim=-1)


        return feat


MODEL_REGISTRY = {

}
def load(model_id: str, freeze: bool = True, cache: str = DEFAULT_CACHE, device: torch.device = "cpu"):
    """
    Download & cache specified model configuration & checkpoint, then load & return module & image processor.

    Note :: We *override* the default `forward()` method of each of the respective model classes with the
            `extract_features` method --> by default passing "NULL" language for any language-conditioned models.
            This can be overridden either by passing in language (as a `str) or by invoking the corresponding methods.
    """
    assert model_id in MODEL_REGISTRY, f"Model ID `{model_id}` not valid, try one of  {list(MODEL_REGISTRY.keys())}"
    print(f'Load Pre-trained model ---> {model_id}')
    # Download Config & Checkpoint (if not in cache)
    # model_cache = Path(cache) / model_id
    config_path, checkpoint_path = Path(cache) / f"{model_id}.json", Path(cache) / f"{model_id}.pt"
    # os.makedirs(model_cache, exist_ok=True)
    assert checkpoint_path.exists() and config_path.exists(), f'{checkpoint_path} or {config_path} model path does not exist'
    # if not checkpoint_path.exists() or not config_path.exists():
    #     gdown.download(id=MODEL_REGISTRY[model_id]["config"], output=str(config_path), quiet=False)
    #     gdown.download(id=MODEL_REGISTRY[model_id]["checkpoint"], output=str(checkpoint_path), quiet=False)

    # Load Configuration --> patch `hf_cache` key if present (don't download to random locations on filesystem)
    with open(config_path, "r") as f:
        model_kwargs = json.load(f)
        # if "hf_cache" in model_kwargs:
        #     model_kwargs["hf_cache"] = str(Path(cache) / "hf-cache")

    # By default, the model's `__call__` method defaults to `forward` --> for downstream applications, override!
    #   > Switch `__call__` to `get_representations`
    # MODEL_REGISTRY[model_id]["cls"].__call__ = MODEL_REGISTRY[model_id]["cls"].get_representations

    # Materialize Model (load weights from checkpoint; note that unused element `_` are the optimizer states...)
    model = MODEL_REGISTRY[model_id]["cls"](**model_kwargs)
    if model_id in ['VMVP']:
        state_dict, _ = torch.load(checkpoint_path, map_location=device)
        emb_dim = model_kwargs['encoder_embed_dim']
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
        emb_dim = model_kwargs['encoder_decoder_cfg']['encoder_embed_dim']
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.replace('module.', '') in model_dict:
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze model parameters if specified (default: True)
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False

    # Build Visual Preprocessing Transform (assumes image is read into a torch.Tensor, but can be adapted)
    if model_id in list(MODEL_REGISTRY.keys()):
        # All models except R3M are by default normalized subject to default IN1K normalization...
        preprocess = T.Compose(
            [
                # T.Resize(model_kwargs["dataset_cfg"]["resolution"]),
                # T.CenterCrop(model_kwargs["resolution"]),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ]
        )
    else:
        raise AttributeError(F'{model_id} dose not exit')

    return model, preprocess, emb_dim
    # return model, emb_dim

def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def save_recons_imgs(
    ori_imgs: torch.Tensor,
    recons_imgs: torch.Tensor,
    save_dir_input: Path,
    identify: str,
    online_normalization
) -> None:
    # import cv2
    import cv2
    # de online transforms function
    def de_online_transform(ori_img, recon_img, norm=online_normalization):
        # rearrange
        ori_imgs = rearrange(ori_img,"c h w -> h w c")
        recon_imgs = rearrange(recon_img, "c h w -> h w c")
        # to Numpy format
        ori_imgs = ori_imgs.detach().numpy()
        recon_imgs = recon_imgs.detach().numpy()
        # DeNormalize
        ori_imgs = np.array(norm[0]) + ori_imgs * np.array(norm[1])
        recon_imgs = np.array(norm[0]) + recon_imgs * np.array(norm[1])
        # to cv format
        ori_imgs = np.uint8(ori_imgs * 255)
        recon_imgs = np.uint8(recon_imgs * 255)

        return ori_imgs, recon_imgs

    save_dir = save_dir_input / identify
    os.makedirs(str(save_dir), exist_ok=True)
    for bid in range(ori_imgs.shape[0]):
        ori_img = ori_imgs[bid]
        recon_img = recons_imgs[bid]
        # de online norm
        ori_img, recon_img = de_online_transform(ori_img, recon_img)

        ori_save_path = save_dir / f"{bid}_raw.jpg"
        recon_save_path = save_dir / f"{bid}_recon.jpg"

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(ori_save_path), ori_img)
        cv2.imwrite(str(recon_save_path), recon_img)