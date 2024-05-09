
import numpy as np

def convert_pose_to_vector(pose):
    vector_pose = []
    for point in pose:
        x, y, z = point
        vector_pose.extend([x, y, z])
    return vector_pose

def scale_and_translate(vector_pose):
    vector_pose = np.array(vector_pose).reshape(-1, 3)
    min_vals = np.min(vector_pose, axis=0)
    max_vals = np.max(vector_pose, axis=0)
    scaler = np.max(max_vals - min_vals)
    return (vector_pose - min_vals) / scaler

def l2_norm(vpose):
    norm = np.linalg.norm(vpose)
    return vpose / norm

def pose_similarity(pose1, pose2):
    vpose1 = convert_pose_to_vector(pose1)
    vpose2 = convert_pose_to_vector(pose2)
    
    vpose1 = scale_and_translate(vpose1).flatten()
    vpose2 = scale_and_translate(vpose2).flatten()
    
    vpose1 = l2_norm(vpose1)
    vpose2 = l2_norm(vpose2)

    return np.linalg.norm(vpose1 - vpose2)

def dtw_similarity(pose_sequence1, pose_sequence2):
    N = len(pose_sequence1)
    K = len(pose_sequence2)

    # Initialize DTW matrix
    dtw_matrix = np.full((N + 1, K + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    # Compute DTW
    for i in range(1, N + 1):
        for j in range(1, K + 1):
            cost = pose_similarity(pose_sequence1[i - 1], pose_sequence2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],      # Insertion
                                        dtw_matrix[i, j - 1],      # Deletion
                                        dtw_matrix[i - 1, j - 1])  # Match

    # Return the DTW distance
    return dtw_matrix[N, K]

if __name__ == '__main__':
    # pose_sequence1 (N, 17, 3)
    # pose_sequence2 (K, 17, 3)
    pose_sequence1 = np.load('data/keypoints/val/sequence_1_3D.npy')
    pose_sequence2 = np.load('data/keypoints/val/sequence_95_3D.npy')
    dtw_score = dtw_similarity(pose_sequence1, pose_sequence2)
    print(dtw_score)