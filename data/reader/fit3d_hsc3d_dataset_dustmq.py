import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from utils.data import flip_data

class FIT3DHSC3D(Dataset):
    def __init__(self, keypoints_root, data_split, n_frames=243, stride=81, w=1920, h=1080, flip=False):
        self.keypoint_root =keypoints_root
        self.data_split = data_split
        self.stride = stride if data_split == 'train' else n_frames
        self.w, self.h = w, h
        self.n_frames = n_frames
        self.flip = flip

        # 데이터 로드
        data_2d, data_3d = self.load_data(keypoints_root, data_split)
        self.data_list_2d, self.data_list_3d, self.data_list_camera = self.split_into_clips(data_2d, data_3d, n_frames, stride)

        assert len(self.data_list_2d) == len(self.data_list_3d)
        assert len(self.data_list_2d) == len(self.data_list_camera)

    def load_data(self, keypoints_root, data_split):
        data_2d = {}
        data_3d = {}

        # 2D 및 3D 키포인트 파일 경로 설정
        path = os.path.join(keypoints_root, data_split)
    
        # 2D 키포인트 로드
        for file in os.listdir(path):
            if file.endswith('_2D.npy'):
                sequence_name = file.split('_')[1]
                file_path = os.path.join(path, file)
                data_2d[sequence_name] = np.load(file_path)

        # 3D 키포인트 로드
        for file in os.listdir(path):
            if file.endswith('_3D.npy'):
                sequence_name = file.split('_')[1]
                file_path = os.path.join(path, file)
                data_3d[sequence_name] = {
                    'keypoints': np.load(file_path),
                    'res_h': self.h,
                    'res_w': self.w
                }

        return data_2d, data_3d

    def split_into_clips(self, data_2d, data_3d, n_frames, stride):
        data_list_2d, data_list_3d, data_list_camera = [], [], []
        for sequence_name in data_2d:
            keypoints_2d = data_2d[sequence_name]
            keypoints_3d = data_3d[sequence_name]['keypoints']
            res_h = data_3d[sequence_name]['res_h']
            res_w = data_3d[sequence_name]['res_w']

            keypoints_2d = self.normalize(keypoints_2d, res_w, res_h)
            keypoints_3d = self.normalize(keypoints_3d, res_w, res_h, is_3d=True)
            keypoints_2d = keypoints_2d[:keypoints_3d.shape[0]]

            clips_2d, clips_3d = self.partition(keypoints_2d, keypoints_3d, n_frames, stride)

            data_list_2d.extend(clips_2d)
            data_list_3d.extend(clips_3d)
            data_list_camera.extend([(res_h, res_w)] * len(clips_2d))

        return data_list_2d, data_list_3d, data_list_camera

    def normalize(keypoints, w, h, is_3d=False):
        result = np.copy(keypoints)
        result[..., :2] = keypoints[..., :2] / w * 2 - [1, h / w]   # for width and height
        if is_3d:
           result[..., 2:] = keypoints[..., 2:] / w * 2   # for depth in 3D keypoints
        return result

    def partition(self, keypoints_2d, keypoints_3d, clip_length, stride):
        if self.data_split == "test":
            stride = clip_length
            
        clips_2d, clips_3d = [], []
        video_length = keypoints_2d.shape[0]
        if video_length <= clip_length:
            new_indices = self.resample(video_length, clip_length)
            clips_2d.append(keypoints_2d[new_indices])
            clips_3d.append(keypoints_3d[new_indices])
        else:
            start_frame = 0
            while (video_length - start_frame) >= clip_length:
                clips_2d.append(keypoints_2d[start_frame:start_frame + clip_length])
                clips_3d.append(keypoints_3d[start_frame:start_frame + clip_length])
                start_frame += stride
            new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
            clips_2d.append(keypoints_2d[new_indices])
            clips_3d.append(keypoints_3d[new_indices])
        return clips_2d, clips_3d

    @staticmethod
    def resample(original_length, target_length):
        """
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68

        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result

    def denormalize(self, keypoints, idx, is_3d=False):
        h, w = self.data_list_camera[idx]
        result = np.copy(keypoints)
        result[..., :2] = (keypoints[..., :2] + np.array([1, h / w])) * w / 2
        if is_3d:
            result[..., 2:] = keypoints[..., 2:] * w / 2
        return result

    def __len__(self):
        return len(self.data_list_2d)

    # def __getitem__(self, idx):
    #     data_2d = torch.tensor(self.data_list_2d[idx], dtype=torch.float32)
    #     data_3d = torch.tensor(self.data_list_3d[idx], dtype=torch.float32)

    #     # 미러링 확률 50%
    #     if random.random() > 0.5:
    #         data_2d = torch.flip(data_2d, [2])  # x축 반전
    #         data_3d = torch.flip(data_3d, [2])

    #     return data_2d, data_3d

    def __getitem__(self, idx):
        data_2d = torch.tensor(self.data_list_2d[idx], dtype=torch.float32)
        data_3d = torch.tensor(self.data_list_3d[idx], dtype=torch.float32)

        if self.data_split == 'val':
            # 테스트 모드에서는 추가 정보 반환
            return data_2d, data_3d

        if self.flip and random.random() > 0.5:
            # 훈련 모드에서 미러링 증강
            data_2d = flip_data(data_2d)
            data_3d = flip_data(data_3d)

        return data_2d, data_3d


# 사용 예시
# dataset = KeypointsDataset('경로/keypoints', 'train')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)