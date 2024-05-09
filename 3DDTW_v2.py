import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    return 1 - cosine(v1, v2)

def dtw_cosine_similarity(pose1, pose2):
    """
    Compute DTW similarity between two sequences of 3D keypoints using cosine similarity.
    pose1 and pose2 are expected to be of shape (N, 17, 3) and (K, 17, 3), respectively.
    """
    N, num_joints, _ = pose1.shape
    K, _, _ = pose2.shape

    # Initialize DTW matrix
    dtw_matrix = np.full((N+1, K+1), float('inf'))
    dtw_matrix[0, 0] = 0

    # Compute DTW with progress display using tqdm
    for i in tqdm(range(1, N+1), desc="Computing DTW", total=N):
        for j in range(1, K+1):
            cost = 0
            for joint in range(num_joints):
                cost += cosine_similarity(pose1[i-1, joint], pose2[j-1, joint])
            cost /= num_joints  # Average cost over all joints

            # Update DTW matrix
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    # Return the normalized DTW distance
    return dtw_matrix[N, K] / (N + K)

# Example usage
# These are dummy poses, replace with actual 3D keypoints
# pose1 = np.random.rand(17, 17, 3)  # Example pose of shape (N, 17, 3)
# pose2 = np.random.rand(20, 17, 3)  # Example pose of shape (K, 17, 3)
pose_sequence1 = np.load('data/keypoints/val/sequence_1_3D.npy')
pose_sequence2 = np.load('data/keypoints/val/sequence_1252_3D.npy')
similarity = dtw_cosine_similarity(pose_sequence1, pose_sequence2)
print(similarity)