import os
import numpy as np
from tqdm import tqdm

from tools import read_json, joint_format_to_h36m ,read_cam_params, _infer_box, camera_to_image_frame, project_point_radial

def save_data_npy(data_root, save_root):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}") # data_root가 없다면 에러 출력
    
    if not os.path.exists(save_root):
        os.makedirs(save_root) # save_root가 없다면 해당 위치에 생성

    subjects = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))] # s01 ,s02..., S12
    cameras = ['50591643', '58860488', '60457274', '65906101']

    total_file_cnt = 0

    for subj in subjects:
        actions_path = os.path.join(data_root, subj, 'joints3d_25') # j3d json path
        actions = [action.split('.')[0] for action in os.listdir(actions_path) if action.endswith('.json')] # action name list

        if not os.path.exists(actions_path):
            print(f"Skipping missing directory: {actions_path}")
            continue

        for action in tqdm(actions, desc=f'Processing {subj}'):
            for cam in cameras:
                cam_path = os.path.join(data_root, subj, 'camera_parameters', cam, f'{action}.json')
                cam_params  = read_cam_params(cam_path)

                action_file = os.path.join(actions_path, f'{action}.json')
                if not os.path.isfile(action_file):
                    print(f"Skipping missing file: {action_file}")
                    continue

                joint_data = read_json(action_file) # load j3d json file
                joint_np = np.array(joint_data['joints3d_25']) # j3d to np array
                joints_3d = joint_format_to_h36m(joint_np)

                joint_3d_images = np.zeros_like(joints_3d)

                for frame, j3d in enumerate(joints_3d):
                    # j3d_cam = world_to_camera_frame(j3d, cam_params)
                    box = _infer_box(j3d, cam_params)
                    j3d_img = camera_to_image_frame(j3d, box, cam_params)
                    joint_3d_images[frame] = j3d_img

                joint_2d_images = np.ones(joint_3d_images.shape)
                joint_2d_images[:,:,:2] = joint_3d_images[:,:,:2]

                total_file_cnt += 1

                joint_2d_path = os.path.join(save_root, f'sequence_{total_file_cnt}_2D.npy')
                joint_3d_path = os.path.join(save_root, f'sequence_{total_file_cnt}_3D.npy')

                np.save(joint_2d_path, joint_2d_images)
                np.save(joint_3d_path, joint_3d_images)

                # return joint_2d_images, joint_3d_images # for check value

def main():
    data_train_root = '../fit3human/train'
    data_val_root = '../fit3human/val'

    save_train_root = '../keypoints/train'
    save_val_root = '../keypoints/val'

    save_data_npy(data_train_root, save_train_root)
    save_data_npy(data_val_root, save_val_root)

if __name__ == '__main__':
    main()
    #j2d, j3d = save_data_npy('../fit3human/train', '../keypoints/train')
    #print(sum(sum(j2d[:,:,:2] != j3d[:,:,:2])))
    # print(j2d)
    # print(j3d)
    #print(f'max x : {np.max(j3d[:,:,0])}')
    #print(f'min x : {np.min(j3d[:,:,0])}')
    #print(f'max y : {np.max(j3d[:,:,1])}')
    #print(f'min y : {np.min(j3d[:,:,1])}')
    #print(f'max z : {np.max(j3d[:,:,2])}')
    #print(f'min z : {np.min(j3d[:,:,2])}')