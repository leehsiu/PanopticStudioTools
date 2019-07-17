""" Util functions for SMPL
@@batch_skew
@@batch_rodrigues
@@batch_lrotmin
@@batch_global_rigid_transformation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as np
import numpy as np
import numpy.linalg as nl


def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape[0]
    col_inds = np.array([1, 2, 3, 5, 6, 7])
    indices = np.reshape(
        np.reshape(np.arange(0, batch_size) * 9, [-1, 1]) + col_inds,
        [-1, 1])
    updates = np.reshape(
        np.stack(
            [
                -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                vec[:, 0]
            ],
            axis=1), [-1])
    out_shape = [batch_size * 9]
    # res = np.scatter_nd(indices, updates, out_shape)
    res = np.zeros(out_shape, dtype=updates.dtype)
    res[indices[:, 0]] = updates
    res = np.reshape(res, [batch_size, 3, 3])

    return res


def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """
    batch_size = theta.shape[0]

    angle = np.expand_dims(nl.norm(theta + 1e-8, axis=1), -1)
    r = np.expand_dims(theta / angle, -1)

    angle = np.expand_dims(angle, -1)
    cos = np.cos(angle)
    sin = np.sin(angle)

    outer = np.matmul(r, np.transpose(r, [0, 2, 1]))

    eyes = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
        r, batch_size=batch_size)
    return R


def batch_lrotmin(theta, name=None):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
    theta = theta[:, 3:]

    # N*23 x 3 x 3
    Rs = batch_rodrigues(np.reshape(theta, [-1, 3]))
    lrotmin = np.reshape(Rs - np.eye(3), [-1, 207])

    return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    nJ = Rs.shape[1]
    N = Rs.shape[0]
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
        rot_x = np.reshape(np.tile(rot_x, [N, 1]), [N, 3, 3])
        root_rotation = np.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = np.expand_dims(Js, -1)

    def make_A(R, t, name=None):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]], 'constant')
        t_homo = np.concatenate([t, np.ones([N, 1, 1])], 1)
        return np.concatenate([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = np.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = np.stack(results, axis=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = np.concatenate([Js, np.zeros([N, nJ, 1, 1])], 2)
    init_bone = np.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]], 'constant')
    A = results - init_bone

    return new_J, A



def batch_global_rigid_transformation_no_root(Rs, Js, parent):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    nJ = Rs.shape[1] + 1
    N = Rs.shape[0]

    # if rotate_base:
    #     print('Flipping the SMPL coordinate frame!!!!')
    #     rot_x = np.array(
    #         [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
    #     rot_x = np.reshape(np.tile(rot_x, [N, 1]), [N, 3, 3])
    #     root_rotation = np.matmul(Rs[:, 0, :, :], rot_x)
    # else:
    #     root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Rs0 = np.eye(3)[None,:,:]

    root_rotation = Rs0.repeat(N,axis=0)
    Js = np.expand_dims(Js, -1)

    def make_A(R, t, name=None):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]], 'constant')
        t_homo = np.concatenate([t, np.ones([N, 1, 1])], 1)
        return np.concatenate([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i-1], j_here)
        res_here = np.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = np.stack(results, axis=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = np.concatenate([Js, np.zeros([N, nJ, 1, 1])], 2)
    init_bone = np.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]], 'constant')
    A = results - init_bone

    return new_J, A


def batch_angleaxis_to_quaternion(theta):
    # convert angle axis to quaternions in a batch
    assert len(theta.shape) == 2 and theta.shape[1] == 3
    ln = nl.norm(theta, axis=1)
    q = np.zeros((theta.shape[0], 4), dtype=theta.dtype)
    q[:, 0] = np.cos(ln / 2)
    ind = np.where(ln > 0)[0]  # indices where length > 0
    q[ind, 1:] = np.sin(ln / 2)[ind, np.newaxis] * theta[ind] / ln[ind, np.newaxis]
    return q


def batch_quaternion_multiply(q1, q2):
    assert q1.shape == q2.shape and len(q1.shape) == 2 and q1.shape[1] == 4
    q = np.zeros_like(q1)
    q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    return q


def batch_quaternion_to_angleaxis(q):
    assert len(q.shape) == 2 and q.shape[1] == 4
    theta = np.zeros((q.shape[0], 3), dtype=q.dtype)
    sin_sq = np.sum(q[:, 1:] ** 2, axis=1)
    ind = np.where(sin_sq > 0)[0]

    sin = np.sqrt(sin_sq)
    cp_sin = np.copy(sin)
    cos = np.copy(q[:, 0])
    cp_sin[cos < 0] *= -1
    cos[cos < 0] *= -1
    angle = 2 * np.arctan2(cp_sin, cos)

    k = angle[ind] / sin[ind]
    theta[ind] = q[ind, 1:] * k[:, np.newaxis]
    return theta
