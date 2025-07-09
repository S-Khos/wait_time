import numpy as np
from fixed_buffer import FixedBuffer
from scipy.interpolate import interp1d

class DBSCAN():

    def __init__(
        self,
        frame_size=None,
        connected_search_radius=60,
        label_search_radius=65,
        neighbor_tol=5,
        buffer_size=100,
        label_start=1,
        x_std_threshold=15
    ) -> None:
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]
        self.raw_matrix = np.zeros((self.frame_width, self.frame_height), dtype=int)
        self.cluster_matrix = np.zeros((self.frame_width, self.frame_height), dtype=int)
        self.connected_search_radius = connected_search_radius
        self.label_search_radius = label_search_radius
        self.neighbor_tol = neighbor_tol
        self.buffer = FixedBuffer(max_len=buffer_size)
        self.output_matrix = None
        self.label_start = label_start
        self.x_std_threshold = x_std_threshold

    def dfs(self, visited, i, j, coordinates) -> None:
        stack = [(i, j)]
        visited[i, j] = True
        # coordinates.append((i, j))
        while stack:
            x, y = stack.pop()
            # Create a grid of offsets
            dx, dy = np.meshgrid(np.arange(-self.connected_search_radius, self.connected_search_radius + 1), 
                                 np.arange(-self.connected_search_radius, self.connected_search_radius + 1))
            dx, dy = dx.flatten(), dy.flatten()
            # Remove the origin point
            valid_offsets = (dx != 0) | (dy != 0)
            dx, dy = dx[valid_offsets], dy[valid_offsets]
            # Calculate neighbor coordinates
            neighbors_x = x + dx
            neighbors_y = y + dy
            # Filter valid neighbor coordinates
            valid_neighbors = (
                (0 <= neighbors_x) & (neighbors_x < self.raw_matrix.shape[0]) &
                (0 <= neighbors_y) & (neighbors_y < self.raw_matrix.shape[1])
            )
            # Apply the boundary check before accessing the arrays
            neighbors_x = neighbors_x[valid_neighbors]
            neighbors_y = neighbors_y[valid_neighbors]
            # Filter valid neighbors
            valid_neighbors = (
                (~visited[neighbors_x, neighbors_y]) &
                (self.raw_matrix[neighbors_x, neighbors_y] == 1)
            )
            valid_neighbors_x = neighbors_x[valid_neighbors]
            valid_neighbors_y = neighbors_y[valid_neighbors]
            # anomaly detection (lane change)
            all_x_coords = np.array(valid_neighbors_x)
            all_x_coords = np.append(all_x_coords, x)
            x_std = np.std(all_x_coords)
            num_neighbors = np.sum(valid_neighbors) 
            if (num_neighbors <= self.neighbor_tol) and (x_std >= self.x_std_threshold):
                self.raw_matrix[x, y] = 0
                # stack.extend(zip(valid_neighbors_x, valid_neighbors_y))
                continue

            # Update visited and coordinates
            visited[valid_neighbors_x, valid_neighbors_y] = True
            coordinates.append((x, y))
            coordinates.extend(zip(valid_neighbors_x, valid_neighbors_y))
            stack.extend(zip(valid_neighbors_x, valid_neighbors_y))

    # def remove_outliers(self, coordinates, z_score_threshold=2.2) -> list:
    #     if len(coordinates) <= 1:
    #         return coordinates
    #     x_coords, y_coords = zip(*coordinates)
    #     x_coords = np.array(x_coords)
    #     y_coords = np.array(y_coords)
    #     z_scores = np.abs((x_coords - np.mean(x_coords)) / np.std(x_coords))
    #     valid_indices = z_scores < z_score_threshold
    #     return list(zip(x_coords[valid_indices], y_coords[valid_indices]))

    def remove_outliers(self, coordinates, z_score_threshold=1.8, window_size=15) -> list:
        if len(coordinates) <= 1:
            return coordinates
        
        # Convert to numpy array and sort by y-coordinate (ascending)
        coords_array = np.array(coordinates)
        sorted_indices = np.argsort(coords_array[:, 1])
        sorted_coords = coords_array[sorted_indices]
        
        valid_mask = np.ones(len(sorted_coords), dtype=bool)
        
        # Apply sliding window z-score filtering
        for i in range(len(sorted_coords)):
            # Define window bounds
            window_start = max(0, i - window_size // 2)
            window_end = min(len(sorted_coords), window_start + window_size)
            
            # Adjust window_start if we're near the end
            if window_end - window_start < window_size and window_end == len(sorted_coords):
                window_start = max(0, window_end - window_size)
            
            # Extract window coordinates
            window_coords = sorted_coords[window_start:window_end]
            x_coords = window_coords[:, 0]
            
            # Calculate z-score for current point relative to window
            if len(x_coords) > 1 and np.std(x_coords) > 0:
                current_x = sorted_coords[i, 0]
                z_score = np.abs((current_x - np.mean(x_coords)) / np.std(x_coords))
                
                # Mark as invalid if z-score exceeds threshold
                if z_score >= z_score_threshold:
                    valid_mask[i] = False
        
        # Filter valid coordinates and return as list of tuples
        valid_coords = sorted_coords[valid_mask]
        return [(int(x), int(y)) for x, y in valid_coords]

    def smooth_trajectory(self, coordinates, base_kernel_size=18, base_std=10, interpolate=False) -> list:
        if len(coordinates) <= 1:
            return coordinates
        # First step: Apply moving average smoothing (existing code)
        num_points = len(coordinates)
        cluster_std = np.std(np.array(coordinates), axis=0)
        # kernal_size = base_kernel_size * (cluster_std[0] / base_std)
        kernal_size = base_kernel_size
        kernel = np.ones(kernal_size) / kernal_size
        x_coords, y_coords = zip(*coordinates)
        padded_x = np.pad(x_coords, pad_width=kernal_size//2, mode='edge')
        padded_y = np.pad(y_coords, pad_width=kernal_size//2, mode='edge')
        smoothed_x = np.convolve(padded_x, kernel, mode='valid')
        smoothed_y = np.convolve(padded_y, kernel, mode='valid')
        
        if interpolate:
            t = np.zeros(len(smoothed_x))
            if len(set(zip(smoothed_x, smoothed_y))) == 1:
                # Just return the single point repeated
                return [(int(smoothed_x[0]), int(smoothed_y[0]))] * num_points
            
            for i in range(1, len(smoothed_x)):
                dx = smoothed_x[i] - smoothed_x[i-1]
                dy = smoothed_y[i] - smoothed_y[i-1]
                t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)
            if t[-1] <= 0:
                # Just return the original coordinates if no meaningful path length
                return [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]
            kind = 'linear'
            
            try:
                fx = interp1d(t, smoothed_x, kind=kind, bounds_error=False, fill_value="extrapolate")
                fy = interp1d(t, smoothed_y, kind=kind, bounds_error=False, fill_value="extrapolate")
                
                t_new = np.linspace(t[0], t[-1], num_points)
                final_x = np.round(fx(t_new)).astype(int)
                final_y = np.round(fy(t_new)).astype(int)

                final_x = np.clip(final_x, 0, self.frame_width - 1)
                final_y = np.clip(final_y, 0, self.frame_height - 1)
                
                return list(zip(final_x, final_y))
            except Exception as e:
                return [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]
        else:
            smoothed_x = np.round(smoothed_x).astype(int)
            smoothed_y = np.round(smoothed_y).astype(int)
            smoothed_coordinates = list(zip(smoothed_x, smoothed_y))
            return smoothed_coordinates

    def fit(self, data) -> int:
        self.buffer.append(data)
        self.raw_matrix = np.zeros((self.frame_width, self.frame_height), dtype=int)
        self.cluster_matrix = np.zeros((self.frame_width, self.frame_height), dtype=int)
        label_coordinates = {}
        label_count = self.label_start
        # flatten combined trajectories into a list of all trajectory points
        vehicle_trajectories = np.concatenate(self.buffer.get_state())
        # filter valid indices
        valid_indices = (
            (0 <= vehicle_trajectories[:, 0])
            & (vehicle_trajectories[:, 0] < self.frame_width)
            & (0 <= vehicle_trajectories[:, 1])
            & (vehicle_trajectories[:, 1] < self.frame_height)
        )
        # Set the valid indices in the matrix to 1
        self.raw_matrix[vehicle_trajectories[valid_indices, 0], vehicle_trajectories[valid_indices, 1]] = 1
        visited = np.full(self.raw_matrix.shape, False, dtype=bool)
        for i in range(len(self.raw_matrix)):
            for j in range(len(self.raw_matrix[0])):
                if self.raw_matrix[i][j] == 1 and not visited[i][j]:
                    coordinates = []
                    # seach for all connected components starting from i, j point
                    self.dfs(visited, i, j, coordinates)
                    #coordinates = self.smooth_trajectory(coordinates)
                    if len(coordinates) >= 30:
                        coordinates = self.remove_outliers(coordinates)
                        label_coordinates[label_count] = (
                            coordinates
                        )
                        coordinates = np.array(coordinates)
                        self.cluster_matrix[coordinates[:, 0], coordinates[:, 1]] = label_count
                        label_count += 1
                    
        return (label_count - 1, label_coordinates)

    def predict(self, centroids):
        labels = np.full(len(centroids), -1)
        for idx, (x, y) in enumerate(centroids):
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                # Determine neighborhood bounds with boundary checking
                x_min = max(0, int(x - self.label_search_radius))
                x_max = min(self.frame_width, int(x + self.label_search_radius + 1))
                y_min = max(0, int(y - self.label_search_radius))
                y_max = min(self.frame_height, int(y + self.label_search_radius + 1))
                
                # Extract neighborhood directly without padding
                neighborhood = self.cluster_matrix[x_min:x_max, y_min:y_max]
                
                # Get non-zero labels
                neighbor_labels = neighborhood[neighborhood != 0]
                
                if neighbor_labels.size > 0:
                    # Use numpy's unique with counts for faster counting
                    unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]
                    labels[idx] = int(most_common_label)
        
        return labels
