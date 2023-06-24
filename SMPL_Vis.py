import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import pyrender
import trimesh
import cv2
from scipy.spatial.transform import Rotation
from pyrender.constants import RenderFlags
from Tools import axes_align

def smpl4vis(smpl_poses,global_rot,source='xsens'):
    aa_representation = []
    for item in smpl_poses:
        item = axes_align(item,source)
        r = Rotation.from_matrix(item)
        aa = r.as_rotvec()
        aa_representation.append(aa)
    global_rot = axes_align(global_rot,source)
    poses = np.asarray([aa_representation])
    global_pose = np.asarray([[Rotation.from_matrix(global_rot).as_rotvec()]])
    poses = torch.tensor(poses,dtype=torch.float32)
    global_pose = torch.tensor(global_pose,dtype=torch.float32)
    return poses, global_pose

class SMPLVis:
    def __init__(self, beta, gender, wireframe=False):
        self.gender = gender
        model_folder =  'C:/Users\gyz95\OneDrive\Desktop\InverseDynamicsPython\human_models'
        self.model = smplx.create(model_folder,
                                  model_type='smpl',
                                  gender=gender,
                                  num_betas=10,
                                  ext='pkl')
        
        self.beta = torch.tensor(beta,dtype=torch.float32)

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=640,
            viewport_height=480,
            point_size=1.0
        )

        self.wireframe = wireframe

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        camera,camera_pose = self.camera_setup()
        self.scene.add(camera, pose=camera_pose)

    def camera_setup(self):
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)  # you can adjust the field of view here
        t = np.array([0, 0, -2.5])  # adjust the z-value for your scene
        #R = [[-1,0,0],[0,1,0],[0,0,-1]]
        theta = 180 * np.pi / 180  # Convert angle to radians
        R = [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t
        return camera, camera_pose

    def smpl_setup(self,pose,global_orient):
        output = self.model(betas=self.beta, return_verts=True, body_pose=pose, global_orient=global_orient)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, self.model.faces, vertex_colors=vertex_colors)
        return tri_mesh

    def create_scene(self,pose,global_orient):
        smpl_mesh = self.smpl_setup(pose,global_orient)
        camera,camera_pose = self.camera_setup()
        mesh = pyrender.Mesh.from_trimesh(smpl_mesh)
        mesh_node = self.scene.add(mesh)
        self.scene.add(camera, pose=camera_pose)
        return mesh_node

    def get_image(self,pose,global_orient):
        mesh_node = self.create_scene(pose,global_orient)
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA
        rgb, _ = self.renderer.render(self.scene)
        self.scene.remove_node(mesh_node)
        return rgb

if __name__ == "__main__":
    beta = [list(np.random.rand(10))]
    pose_xsens = np.load('C:/Users\gyz95\Downloads/angles.npy')
    xsens_rotm = []
    for i in range(len(pose_xsens[0])):
        xsens_rotm.append(pose_xsens[0][i])
    global_rot = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    xsens_rotm.insert(0, list(global_rot))
    aa_representation = []
    for item in xsens_rotm:
        r = Rotation.from_matrix(item)
        aa = r.as_rotvec()
        aa_representation.append(aa)
    poses = [aa_representation[1:]]
    global_pose = [[aa_representation[0]]]
    poses = torch.tensor(poses,dtype=torch.float32)
    global_pose = torch.tensor(global_pose,dtype=torch.float32)
    vis = SMPLVis(beta,'male')
    scene = vis.create_scene(poses,global_pose)
    image = vis.get_image(scene)
    cv2.imshow('scene',image)
    cv2.waitKey(0)
