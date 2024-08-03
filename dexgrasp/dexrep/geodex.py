
import torch
from pytorch3d.transforms import quaternion_to_matrix

import trimesh
import numpy as np
import point_cloud_utils as pcu

from numpy.random import RandomState

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeoDexWrapper:
    def __init__(self, args):
        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        self.load_env = []
        self.rand = RandomState(42)

        if "geodex" in args["env"].keys():
            self.sample_method = args["env"]["geodex"]["sample_method"]
            self.sample_num_points = args["env"]["geodex"]["sample_num_points"]
        else:
            raise KeyError(f"sample_method not impletement")

    def _normalize_points(self, point_set):
        """zero-center and scale to unit sphere"""
        # zero-center the points
        mean = torch.mean(point_set, dim=0)
        point_set_centered = point_set - mean
        # calculate the maximum distance from the points to the origin (after centering)
        dist = torch.max(torch.norm(point_set_centered, p=2, dim=1))
        # scale the points to be on the unit sphere
        point_set_normalized = point_set_centered / dist

        return point_set_normalized

    def load_cache_stl_file(self, obj_path, obj_idx, scale=1.0):
        obj_mesh = trimesh.load(str(obj_path))
        # scale mesh
        obj_mesh = obj_mesh.apply_scale(scale)
        # sample points and normals
        if self.sample_method == "average":
            object_verts = np.array(obj_mesh.vertices)
            object_normals = np.array(obj_mesh.vertex_normals)
            object_faces = np.array(obj_mesh.faces)
            fid, bc = pcu.sample_mesh_random(object_verts, object_faces, self.sample_num_points)
            sampled_points = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_verts)
            sampled_normals = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_normals)
            # add to pcd list
            self.scaled_sampled_points[obj_idx] = sampled_points
            self.scaled_sampled_normals[obj_idx] = sampled_normals
        elif self.sample_method == "vert":
            self.scaled_sampled_points[obj_idx] = np.array(obj_mesh.vertices)
            self.scaled_sampled_normals[obj_idx] = np.array(obj_mesh.vertex_normals)
        else:
            raise KeyError(f"sample_method <{self.sample_method}> not impleted!")


    def load_batch_env_obj(self, env_obj_idx):
        if self.sample_method == "average":
            # load env points and normals
            self.obs_batch_obj_points.append(
                np.copy(self.scaled_sampled_points[env_obj_idx])
            )
            self.obs_batch_obj_normals.append(
                np.copy(self.scaled_sampled_normals[env_obj_idx])
            )
        elif self.sample_method == 'vert':
            self.load_env.append(env_obj_idx)
        else:
            raise KeyError(f"sample_method <{self.sample_method}> not impleted!")

    def pre_observation(self, obj_rot, goal_rot, clip_range):
        if self.sample_method == "vert":
            for env_idx in self.load_env:
                selected = self.rand.randint(
                    low=0, high=self.scaled_sampled_points[env_idx].shape[0], size=self.sample_num_points)
                sampled_points = np.copy(self.scaled_sampled_points[env_idx][selected])
                sampled_normals = np.copy(self.scaled_sampled_normals[env_idx][selected])
                self.obs_batch_obj_points.append(sampled_points)
                self.obs_batch_obj_normals.append(sampled_normals)

        if isinstance(self.obs_batch_obj_points, np.ndarray) or isinstance(self.obs_batch_obj_points, list):
            self.obs_batch_obj_points = torch.tensor(np.array(self.obs_batch_obj_points), dtype=torch.float32).to(
                device)
            self.obs_batch_obj_normals = torch.tensor(np.array(self.obs_batch_obj_normals), dtype=torch.float32).to(
                device)

        rot_obj = quaternion_to_matrix(torch.roll(obj_rot, 1, dims=1))
        rot_goal = quaternion_to_matrix(torch.roll(goal_rot, 1, dims=1))

        object_points = torch.matmul(torch.clone(self.obs_batch_obj_points), rot_obj.transpose(-1, -2))
        # object_points = self._normalize_points(object_points)
        object_norms = torch.matmul(torch.clone(self.obs_batch_obj_normals), rot_obj.transpose(-1, -2))

        target_points = torch.matmul(torch.clone(self.obs_batch_obj_points), rot_goal.transpose(-1, -2))
        # target_points = self._normalize_points(target_points)
        target_norms = torch.matmul(torch.clone(self.obs_batch_obj_normals), rot_goal.transpose(-1, -2))

        geodex_feat = torch.cat([object_points.reshape(rot_obj.size(0), -1),
                                 object_norms.reshape(rot_obj.size(0), -1),
                                 target_points.reshape(rot_obj.size(0), -1),
                                 target_norms.reshape(rot_obj.size(0), -1)], dim=-1)
        if self.sample_method == "vert":
            del self.obs_batch_obj_points
            del self.obs_batch_obj_normals
            self.obs_batch_obj_points = []
            self.obs_batch_obj_normals = []

        return geodex_feat