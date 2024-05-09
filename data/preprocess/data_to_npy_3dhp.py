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

def project_3d_to_2d(joints3d, intrinsics):
    p = intrinsics['p'][:, [1, 0]]
    x = joints3d[:, :2] / joints3d[:, 2:3]
    r2 = np.sum(x**2, axis=1)
    radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
    tan = np.matmul(x, np.transpose(p))
    xx = x*(tan + radial) + r2[:, np.newaxis] * p
    proj = intrinsics['f'] * xx + intrinsics['c']
    return proj

def save_3d_npy(data_root, save_root, data_name):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    sub_folder_lst = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    
    total_file_count = 0  # 전체 파일 카운터

    for sub_folder in sub_folder_lst:
        json_path = os.path.join(data_root, sub_folder, 'joints3d_25')
        
        if not os.path.exists(json_path):
            print(f"Skipping missing directory: {json_path}")
            continue

        json_lst = [file for file in os.listdir(json_path) if file.endswith('.json')]
        
        for json_file in tqdm(json_lst, desc=f'Processing {sub_folder}'):
            joint_file = os.path.join(json_path, json_file)
            if not os.path.isfile(joint_file):
                print(f"Skipping missing file: {joint_file}")
                continue

            joint_data = read_json(joint_file)
            joint_np = np.array(joint_data['joints3d_25'])
            joints = joint_format_to_h36m(joint_np)

            for j in range(4):
                total_file_count += 1  # 파일 카운터 증가
                file_path = os.path.join(save_root, f'{data_name}_sequence_{total_file_count}_3D.npy')
                np.save(file_path, joints)

def save_2d_npy(data_root, save_root, data_name):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    subjects = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    cameras = ['50591643', '58860488', '60457274', '65906101']

    total_file_count = 0  # 전체 파일 카운터

    max_x, max_y = -np.inf, -np.inf
    min_x, min_y = np.inf, np.inf

    for subj in subjects:
        actions_path = os.path.join(data_root, subj, 'joints3d_25')
        actions = [action.split('.')[0] for action in os.listdir(actions_path) if action.endswith('.json')]
        
        for action in tqdm(actions, desc=f'Processing {subj}'):
            for cam in cameras:
                cam_path = os.path.join(data_root, subj, 'camera_parameters', cam, f'{action}.json')
                j3d_path = os.path.join(data_root, subj, 'joints3d_25', f'{action}.json')

                if not os.path.exists(cam_path) or not os.path.exists(j3d_path):
                    continue

                cam_params = read_cam_params(cam_path)
                with open(j3d_path) as f:
                    j3ds = np.array(json.load(f)['joints3d_25'])

                all_frames_2d = np.zeros((len(j3ds), 25, 2))
                for frame_id, j3d in enumerate(j3ds):
                    j2d = project_3d_to_2d(j3d, cam_params['intrinsics_w_distortion'])
                    all_frames_2d[frame_id] = j2d

                # H36M 포맷으로 변경
                all_frames_2d = joint_format_to_h36m(all_frames_2d) 
                # Confidence score 추가
                confidence_scores = np.ones((all_frames_2d.shape[0], all_frames_2d.shape[1], 1))
                all_frames_2d_with_confidence = np.concatenate((all_frames_2d, confidence_scores), axis=-1)

                for frame in all_frames_2d_with_confidence:
                    # 최대값들 갱신
                    max_x = max(max_x, np.max(frame[:, 0]))
                    max_y = max(max_y, np.max(frame[:, 1]))

                    # 최소값들 갱신 
                    min_x = min(min_x, np.min(frame[:, 0]))
                    min_y = min(min_y, np.min(frame[:, 1]))

                total_file_count += 1  # 파일 카운터 증가
                output_path = os.path.join(save_root, f'{data_name}_sequence_{total_file_count}_2D.npy')
                # np.save(output_path, all_frames_2d_with_confidence)
    print(f'x:[{min_x},{max_x}]')
    print(f'x:[{min_y},{max_y}]')

def main():
    fit3d_train_root = '../fit3d/train'
    fit3d_val_root = '../fit3d/val'

    humansc3d_train_root = '../humansc3d/train'
    humansc3d_val_root = '../humansc3d/val'
    
    save_train_root = '../keypoints/train'
    save_val_root = '../keypoints/val'

    save_2d_npy(fit3d_train_root, save_train_root, 'fit3d')
    # save_3d_npy(fit3d_train_root, save_train_root, 'fit3d')
    # save_2d_npy(fit3d_val_root, save_val_root, 'fit3d')
    # save_3d_npy(fit3d_val_root, save_val_root, 'fit3d')

    save_2d_npy(humansc3d_train_root, save_train_root, 'humansc3d')
    # save_3d_npy(humansc3d_train_root, save_train_root, 'humansc3d')
    # save_2d_npy(humansc3d_val_root, save_val_root, 'humansc3d')
    # save_3d_npy(humansc3d_val_root, save_val_root, 'humansc3d')

if __name__ == '__main__':
    main()