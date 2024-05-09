import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def joint_format_to_h36m(joints):
    joints_crop = joints[:, :17, :]
    idx_pairs = [(11, 14), (12, 15), (13, 16)]
    
    for idx1, idx2 in idx_pairs:
        joints_crop[:, [idx1, idx2]] = joints_crop[:, [idx2, idx1]]

    return joints_crop

def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2])
    return cam_params

def _infer_box(pose3d, camera_params, rootIdx=0):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 900.0
    br_joint = root_joint.copy()
    br_joint[:2] += 900.0

    # project_point_radial 함수를 사용하여 2D 투영
    tl2d = project_point_radial(np.array([tl_joint]), camera_params)
    br2d = project_point_radial(np.array([br_joint]), camera_params)

    return np.array([tl2d[0][0], tl2d[0][1], br2d[0][0], br2d[0][1]])

def camera_to_image_frame(pose3d, box, camera_params, rootIdx=0):
    rectangle_3d_size = 1800.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size

    # project_point_radial 함수를 사용하여 2D 투영
    proj3d = project_point_radial(pose3d, camera_params)

    pose3d_image_frame = np.zeros_like(proj3d)
    pose3d_image_frame[:, :2] = proj3d[:, :2]
    pose3d_depth = ratio * (proj3d[:, 2] - proj3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth

    return pose3d_image_frame

def project_point_radial(points, camera_params):
    # Extract and transpose camera parameters
    R = camera_params['extrinsics']['R']
    T = camera_params['extrinsics']['T']
    f = camera_params['intrinsics_w_distortion']['f'][0].T
    c = camera_params['intrinsics_w_distortion']['c'][0].T
    k = camera_params['intrinsics_w_distortion']['k'][0].T
    p = camera_params['intrinsics_w_distortion']['p'][0].T

    # Ensure points is a matrix of 3-dimensional points
    assert len(points.shape) == 2
    assert points.shape[1] == 3

    N = points.shape[0]
    X = np.matmul(points - T, R.T)  # rotate and translate
    XX = X[:, :2] / X[:, 2].reshape(-1, 1)
    r2 = XX[:, 0]**2 + XX[:, 1]**2

    radial = 1 + k[0]*r2 + k[1]*r2**2 + k[2]*r2**3
    tan = p[0]*XX[:, 1] + p[1]*XX[:, 0]

    XXX = XX * radial.reshape(-1, 1) + np.array([p[1], p[0]])*r2.reshape(-1, 1)

    Proj = np.zeros((N, 2))
    Proj[:, 0] = f[0] * XXX[:, 0] + c[0]
    Proj[:, 1] = f[1] * XXX[:, 1] + c[1]

    D = X[:, 2]

    Proj3D = np.hstack((Proj, D.reshape(-1, 1)))

    return Proj3D

