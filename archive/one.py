import numpy as np
from sklearn.mixture import GaussianMixture

def dfs(matrix, visited, i, j, label, coordinates):
    # Directions for 8 neighbors (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    stack = [(i, j)]
    visited[i][j] = True
    matrix[i][j] = label
    coordinates.append((i, j))  # Store the coordinates of this component
    
    while stack:
        x, y = stack.pop()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and not visited[nx][ny] and matrix[nx][ny] == 1:
                visited[nx][ny] = True
                matrix[nx][ny] = label
                coordinates.append((nx, ny))
                stack.append((nx, ny))

def find_connected_components(matrix):
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    label = 2  # Start labeling components from 2 (as 1 is already used in the input)
    label_coordinates = {}  # Dictionary to store coordinates for each label
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1 and not visited[i][j]:
                coordinates = []  # List to store the coordinates of the current component
                dfs(matrix, visited, i, j, label, coordinates)
                label_coordinates[label] = coordinates  # Store the coordinates for this label
                label += 1  # Increment label for the next component
    
    return matrix, label_coordinates

def calculate_mean(label_coordinates):
    means = {}
    for label, coordinates in label_coordinates.items():
        # Calculate mean of row and column indices for each label
        mean_row = sum(x for x, y in coordinates) / len(coordinates)
        mean_col = sum(y for x, y in coordinates) / len(coordinates)
        means[label] = (mean_row, mean_col)
    
    return means

# Input Matrix
input_matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0]
]

print("Before:")
for row in input_matrix:
    print(row)

# Step 1: Find connected components and store their coordinates
output_matrix, label_coordinates = find_connected_components(input_matrix)

print("\nAfter:")
for row in output_matrix:
    print(row)

# Step 2: Calculate the mean for each label
means = calculate_mean(label_coordinates)

print("\nMeans of each label:")
for label, (mean_row, mean_col) in means.items():
    print(f"Label {label}: Mean Row = {mean_row:.2f}, Mean Column = {mean_col:.2f}")

# Step 3: Prepare the data for GMM
# Convert the coordinates into a 2D array where each row is a [row, col] pair
data_points = np.array([coord for coords in label_coordinates.values() for coord in coords])

# Step 4: Initialize GMM with the means of connected components as the initial centroids
initial_means = np.array(list(means.values()))  # This will be of shape (n_components, 2)

gmm = GaussianMixture(n_components=len(means), means_init=initial_means)

# Step 5: Fit the GMM to the data (coordinates of connected components)
gmm.fit(data_points)

# Step 6: Output the GMM results
print("\nGMM Means after fitting:")
print(gmm.means_)

print("\nGMM Weights:")
print(gmm.weights_)

print("\nGMM Covariances:")
print(gmm.covariances_)

