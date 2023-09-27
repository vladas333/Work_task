import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy


# Define Octree Node class
class OctreeNode:
    def __init__(self, min_coords, max_coords):
        self.min_coords = min_coords
        self.max_coords = max_coords
        self.children = []

def create_octree(point_cloud, min_coords, max_coords, depth):
    if depth == 0:
        return None

    node = OctreeNode(min_coords, max_coords)
    
    # Embed a sphere in the current node
    sphere_radius = np.linalg.norm(np.array(max_coords) - np.array(min_coords)) / 2.0
    sphere_center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
    
    # Subdivide into 8 child nodes
    mid_x = (min_coords[0] + max_coords[0]) / 2.0
    mid_y = (min_coords[1] + max_coords[1]) / 2.0
    mid_z = (min_coords[2] + max_coords[2]) / 2.0

    node.children.append(create_octree(point_cloud, min_coords, [mid_x, mid_y, mid_z], depth - 1))
    node.children.append(create_octree(point_cloud, [mid_x, min_coords[1], min_coords[2]], [max_coords[0], mid_y, mid_z], depth - 1))
    node.children.append(create_octree(point_cloud, [min_coords[0], mid_y, min_coords[2]], [mid_x, max_coords[1], mid_z], depth - 1))
    node.children.append(create_octree(point_cloud, [mid_x, mid_y, min_coords[2]], [max_coords[0], max_coords[1], mid_z], depth - 1))
    node.children.append(create_octree(point_cloud, [min_coords[0], min_coords[1], mid_z], [mid_x, mid_y, max_coords[2]], depth - 1))
    node.children.append(create_octree(point_cloud, [mid_x, min_coords[1], mid_z], [max_coords[0], mid_y, max_coords[2]], depth - 1))
    node.children.append(create_octree(point_cloud, [min_coords[0], mid_y, mid_z], [mid_x, max_coords[1], max_coords[2]], depth - 1))
    node.children.append(create_octree(point_cloud, [mid_x, mid_y, mid_z], max_coords, depth - 1))

    return node

# Load LAS file
las_path = "2743_1234.las"
las_data = laspy.read(las_path)

# Extract point cloud data
point_cloud = np.vstack((las_data.x, las_data.y, las_data.z)).T

# Define the bounding box for the entire point cloud
min_coords = [np.min(point_cloud[:, 0]), np.min(point_cloud[:, 1]), np.min(point_cloud[:, 2])]
max_coords = [np.max(point_cloud[:, 0]), np.max(point_cloud[:, 1]), np.max(point_cloud[:, 2])]

# Set the depth for Octree
depth = 900

# Create Octree
octree_root = create_octree(point_cloud, min_coords, max_coords, depth)

# Visualize the Octree and embedded spheres using Matplotlib
def plot_octree(node, ax):
    if node is not None:
        min_coords = node.min_coords
        max_coords = node.max_coords
        
        # Plot the embedded sphere
        sphere_radius = np.linalg.norm(np.array(max_coords) - np.array(min_coords)) / 2.0
        sphere_center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
        ax.scatter(*sphere_center, c='red', s=5)
        
        # Recursively plot child nodes
        for child in node.children:
            plot_octree(child, ax)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='blue', alpha=0.5)

# Plot the Octree with embedded spheres
plot_octree(octree_root, ax)

# Set plot limits
ax.set_xlim(min_coords[0], max_coords[0])
ax.set_ylim(min_coords[1], max_coords[1])
ax.set_zlim(min_coords[2], max_coords[2])

# Save to file
# plt.savefig("octree_with_spheres.png", dpi=300, bbox_inches='tight')
# Show the plot
# plt.show()

# Define a function to save a cube as an image
def save_cube_image(min_coords, max_coords, sphere_center, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='blue', alpha=0.5)
    
    # Plot the Octree cube with embedded sphere
    ax.scatter(*sphere_center, c='red', s=5)
    
    # Set plot limits for the cube
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])

    # Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Create and save images for each cube
def save_all_cube_images(node, filename_prefix):
    if node is not None:
        min_coords = node.min_coords
        max_coords = node.max_coords
        sphere_center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
        
        # Save the cube image
        save_cube_image(min_coords, max_coords, sphere_center, f"{filename_prefix}_cube.png")
        
        # Recursively save child nodes
        for i, child in enumerate(node.children):
            save_all_cube_images(child, f"{filename_prefix}_child{i}")

# Save images for all eight cubes starting from the root
# save_all_cube_images(octree_root, "cube")

# Show the plot for the entire Octree (optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='blue', alpha=0.5)
plot_octree(octree_root, ax)
ax.set_xlim(min_coords[0], max_coords[0])
ax.set_ylim(min_coords[1], max_coords[1])
ax.set_zlim(min_coords[2], max_coords[2])
plt.show()