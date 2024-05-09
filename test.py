# from data.reader.h36m import DataReaderH36M

# data_reader = DataReaderH36M(n_frames=243, sample_stride=1, data_stride_train=1,
#                             data_stride_test=1, read_confidence=True,
#                             dt_root='data/motion3d', dt_file='h36m_sh_conf_cam_source_final.pkl')

# data_reader.read_2d()
# data_reader.read_3d()

from data.reader.fit3d_hsc3d_dataset import FitHscDataset3D

dataset = FitHscDataset3D(keypoints_path='data/keypoints', data_split='train', flip=True)
for i in range(len(dataset)):
        dataset[i]

