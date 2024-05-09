import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# def project_3d_to_2d(joints3d, intrinsics):
#     p = intrinsics['p'][:, [1, 0]]
#     x = joints3d[:, :2] / joints3d[:, 2:3]
#     r2 = np.sum(x**2, axis=1)
#     radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
#     tan = np.matmul(x, np.transpose(p))
#     xx = x*(tan + radial) + r2[:, np.newaxis] * p
#     proj = intrinsics['f'] * xx + intrinsics['c']
#     return proj

####### LCN 코드

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 900.0
    br_joint = root_joint.copy()
    br_joint[:2] += 900.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    fx, fy = camera['intrinsics_w_distortion']['f'][0][0], camera['intrinsics_w_distortion']['f'][0][1]
    cx, cy = camera['intrinsics_w_distortion']['c'][0][0], camera['intrinsics_w_distortion']['c'][0][1]

    tl2d = _weak_project(tl_joint, fx, fy, cx, cy).flatten()
    br2d = _weak_project(br_joint, fx, fy, cx, cy).flatten()

    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])

def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

def world_to_camera_frame(P, R, T):
    assert len(P.shape) == 2
    assert P.shape[1] == 3
    X_cam = R.dot(P.T - T.T) # rotate and translate

    return X_cam.T

def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 1800.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    
    fx, fy = camera['intrinsics_w_distortion']['f'][0][0], camera['intrinsics_w_distortion']['f'][0][1]
    cx, cy = camera['intrinsics_w_distortion']['c'][0][0], camera['intrinsics_w_distortion']['c'][0][1]

    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(pose3d.copy(), fx, fy, cx, cy)
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth

    return pose3d_image_frame

def save_3d_npy(data_root, save_root):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    subjects = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))] # s01 ,s02...
    cameras = ['50591643', '58860488', '60457274', '65906101']

    total_file_count = 0  # 전체 파일 카운터

    for subj in subjects:
        actions_path = os.path.join(data_root, subj, 'joints3d_25') # 3d point 값이 있는 폴더 주소
        actions = [action.split('.')[0] for action in os.listdir(actions_path) if action.endswith('.json')] # 폴더에서 파일명만 가져오기
        
        if not os.path.exists(actions_path):
            print(f"Skipping missing directory: {actions_path}")
            continue
        
        for action in tqdm(actions, desc=f'Processing 3D {subj}'):
            for cam in cameras:
                cam_path = os.path.join(data_root, subj, 'camera_parameters', cam, f'{action}.json')
                cam_params  = read_cam_params(cam_path)

                action_file = os.path.join(actions_path, action+'.json')
                if not os.path.isfile(action_file):
                    print(f"Skipping missing file: {action_file}")
                    continue

                joint_data = read_json(action_file)
                joint_np = np.array(joint_data['joints3d_25'])
                joints = joint_format_to_h36m(joint_np)
                
                R = cam_params['extrinsics']['R']
                T = cam_params['extrinsics']['T']

                joint_3d_images = np.zeros_like(joints) 
                for frame, joint in enumerate(joints):
                    joint_cam = world_to_camera_frame(joint, R, T) # world cood to camera cood
                    box = _infer_box(joint_cam, cam_params, 0) # joint 좌표, camera 파라미터, root 좌표 (기본값 0이고 그냥 0으로 하면 될듯)
                    joint_3d_image = camera_to_image_frame(joint, box, cam_params, 0)
                    joint_3d_images[frame] = joint_3d_image

                total_file_count += 1  # 파일 카운터 증가
                file_path = os.path.join(save_root, f'sequence_{total_file_count}_3D.npy')
                return joint_3d_images
                # np.save(file_path, joint_3d_images)

def save_2d_npy(data_root, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    subjects = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    cameras = ['50591643', '58860488', '60457274', '65906101']

    total_file_count = 0  # 전체 파일 카운터

    for subj in subjects:
        actions_path = os.path.join(data_root, subj, 'joints3d_25')
        actions = [action.split('.')[0] for action in os.listdir(actions_path) if action.endswith('.json')]
        
        for action in tqdm(actions, desc=f'Processing 2D {subj}'):
            for cam in cameras:
                cam_path = os.path.join(data_root, subj, 'camera_parameters', cam, f'{action}.json')
                j3d_path = os.path.join(data_root, subj, 'joints3d_25', f'{action}.json')

                if not os.path.exists(cam_path) or not os.path.exists(j3d_path):
                    continue

                cam_params = read_cam_params(cam_path)
                R = cam_params['extrinsics']['R']
                T = cam_params['extrinsics']['T']
                
                joint_data = read_json(j3d_path)
                joint_np = np.array(joint_data['joints3d_25'])
                joints = joint_format_to_h36m(joint_np)

                all_frames_2d = np.zeros((joints.shape[0], 17, 3))
                for frame_id, j3d in enumerate(joints):
                    j3d_cam = world_to_camera_frame(j3d, R, T)
                    j2d = project_3d_to_2d(j3d_cam, cam_params['intrinsics_w_distortion'])
                    box = _infer_box(j3d_cam, cam_params, 0) # joint 좌표, camera 파라미터, root 좌표 (기본값 0이고 그냥 0으로 하면 될듯)
                    joint_3d_image = camera_to_image_frame(j3d_cam, box, cam_params, 0)
                    all_frames_2d[frame_id] = j2d

                # Confidence score 추가
                confidence_scores = np.ones((all_frames_2d.shape[0], all_frames_2d.shape[1], 1))
                all_frames_2d_with_confidence = np.concatenate((all_frames_2d, confidence_scores), axis=-1)

                total_file_count += 1  # 파일 카운터 증가
                output_path = os.path.join(save_root, f'sequence_{total_file_count}_2D.npy')
                np.save(output_path, all_frames_2d_with_confidence)


def main():
    data_train_root = '../fit3human/train'
    data_val_root = '../fit3human/val'

    save_train_root = '../keypoints/train'
    save_val_root = '../keypoints/val'

    # save_2d_npy(data_train_root, save_train_root)
    # save_3d_npy(data_train_root, save_train_root)

    # save_2d_npy(data_val_root, save_val_root)
    save_3d_npy(data_val_root, save_val_root)

if __name__ == '__main__':
    # main()
    j3d = save_3d_npy('../fit3human/val', '../keypoints/val')
    print(f'max x : {np.max(j3d[:,:,0])}')
    print(f'min x : {np.min(j3d[:,:,0])}')
    print(f'max y : {np.max(j3d[:,:,1])}')
    print(f'min y : {np.min(j3d[:,:,1])}')
    print(f'max z : {np.max(j3d[:,:,2])}')
    print(f'min z : {np.min(j3d[:,:,2])}')