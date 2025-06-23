

# def plot_clusters(X, labels, centroids):
#     unique_labels = np.unique(labels)
#     plt.figure(figsize=(8, 6))
    
#     # Convert list of tuples to NumPy array for plotting
#     X_array = np.array(X)

#     # Plot each cluster
#     for label in unique_labels:
#         cluster_points = X_array[labels == label]
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

#     # Plot centroids
#     plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    
#     plt.title('K-Means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.grid(True)
#     plt.show()



# def transfer_data_to_powerbi(data):
#     powerbi_url = "https://api.powerbi.com/beta/bf06f9ba-7005-4119-86fc-96e4fcd96d6d/datasets/1ddd1a34-8ed9-4b51-b827-17b246402992/rows?experience=power-bi&key=0XpfKIGf1DIEqKlw37RTpTnKI2h%2F9oqJ6sFviflhf%2BKeQr96ZSJFNnlf0r7VUbsTLCPh5T6rkKhvw8wlu9sYmQ%3D%3D"
#     # Send the data to Power BI

#     response = requests.post(powerbi_url, json=data, verify=certifi.where())

#     # Check the response
#     if response.status_code == 200:
#         print("Data sent successfully!")
#     else:
#         print(f"Error: {response.status_code} - {response.text}")

    # kmeans = KMeans(n_clusters=2, max_iter=15)
    # print(lane_extraction.all_traj)
    # labels = kmeans.fit_predict(lane_extraction.all_traj)
    # centroids = kmeans.get_centroids()
    # plot_clusters(lane_extraction.all_traj, labels, centroids)


import requests
import pip_system_certs.wrapt_requests
powerbi_url = "https://api.powerbi.com/beta/bf06f9ba-7005-4119-86fc-96e4fcd96d6d/datasets/1ddd1a34-8ed9-4b51-b827-17b246402992/rows?experience=power-bi&key=0XpfKIGf1DIEqKlw37RTpTnKI2h%2F9oqJ6sFviflhf%2BKeQr96ZSJFNnlf0r7VUbsTLCPh5T6rkKhvw8wlu9sYmQ%3D%3D"
response = requests.get(powerbi_url)
print(response.status_code)