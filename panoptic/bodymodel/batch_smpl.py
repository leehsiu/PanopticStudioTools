"""
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# import tensorflow as tf
from fullycapture.model.batch_lbs import batch_rodrigues, batch_global_rigid_transformation,batch_global_rigid_transformation_no_root

#convert chumpy objects to numpy
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class batchSMPLNumpy(object):
    def __init__(self, pkl_path=os.path.join(dir_path, 'neutral_smpl_with_cocoplus_reg.pkl'), reg_type='total', dtype=np.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'r') as f:
            dd = pickle.load(f)
        # Mean template verticesq
        self.v_template = undo_chumpy(dd['v_template'])
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = shapedir

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = np.array(dd['J_regressor'].T.todense())

        #self.J_template = self.J_regressor.T.dot(self.v_template)
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = posedirs

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = undo_chumpy(dd['weights']),


        if reg_type=='total':
            self.joint_regressor = np.array(dd['J_regressor_total'].todense().T)
        else:
            self.joint_regressor = self.J_regressor
        self.f = dd['f']

        self.nJoints = self.J_regressor.shape[1]
        self.J_template = self.J_regressor.T.dot(self.v_template)
        
    def __call__(self, beta, theta, reg_type='total', name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890 x 3
        v_shaped = np.reshape(
            np.matmul(beta, self.shapedirs),
            [-1, self.size[0], self.size[1]]) + self.v_template

        # 2. Infer shape-dependent joint locations.
        Jx = np.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = np.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = np.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = np.stack([Jx, Jy, Jz], axis=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = np.reshape(
            batch_rodrigues(np.reshape(theta, [-1, 3])), [-1, self.nJoints, 3, 3])
        # Ignore global rotation.
        pose_feature = np.reshape(Rs[:, 1:, :, :] - np.eye(3),
                                  [-1, (self.nJoints*3-3)*3])

        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = np.reshape(
            np.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

        # 5. Do skinning:
        # W is N x 6890 x 24
        W = np.reshape(
            np.tile(self.weights, [num_batch, 1]), [num_batch, -1, self.nJoints])
        # (N x 6890 x 24) x (N x 24 x 16)
        T = np.reshape(
            np.matmul(W, np.reshape(A, [num_batch, self.nJoints, 16])),
            [num_batch, -1, 4, 4])
        v_posed_homo = np.concatenate(
            [v_posed, np.ones([num_batch, v_posed.shape[1], 1])], 2)
        v_homo = np.matmul(T, np.expand_dims(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]


        #print(self.joint_regressor.shape)
        # Get cocoplus or lsp joints:
        joint_x = np.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = np.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = np.matmul(verts[:, :, 2], self.joint_regressor)
        #print(joint_x.shape)
        joints = np.stack([joint_x, joint_y, joint_z], axis=2)
        #print(joints.shape)
        return verts, joints
        

if __name__ == '__main__':
    smpl = batchSMPLNumpy()
    fileName = '/media/posefs3b/Users/donglaix/mosh_data/neutrMosh/neutrSMPL_CMU/01/01_01.pkl'
    with open(fileName, 'rb') as f:
        MoshParam = pickle.load(f)

    batch_size = MoshParam['trans'].shape[0]
    betas = MoshParam['betas']
    batch_betas = np.tile(betas, (batch_size, 1))

    verts, joints = smpl(batch_betas, MoshParam['poses'], get_skin=True)
    verts += np.expand_dims(MoshParam['trans'], axis=1)
