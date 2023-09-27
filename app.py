import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy


class OctreeNode:
    def __init__(self, min_coords, max_coords):
        self.min_coords = min_coords
        self.max_coords = max_coords
        self.children = []

def create_octree(point_cloud, min_coords, max_coords, depth):
    if depth == 0:
        return None

    node = OctreeNode(min_coords, max_coords)
    sphere_radius = np.linalg.norm(np.array(max_coords) - np.array(min_coords)) / 2.0
    sphere_center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
    
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

las_path = "2743_1234.las"
las_data = laspy.read(las_path)

point_cloud = np.vstack((las_data.x, las_data.y, las_data.z)).T

min_coords = [np.min(point_cloud[:, 0]), np.min(point_cloud[:, 1]), np.min(point_cloud[:, 2])]
max_coords = [np.max(point_cloud[:, 0]), np.max(point_cloud[:, 1]), np.max(point_cloud[:, 2])]

depth = 900
octree_root = create_octree(point_cloud, min_coords, max_coords, depth)

def plot_octree(node, ax):
    if node is not None:
        min_coords = node.min_coords
        max_coords = node.max_coords
        
        sphere_radius = np.linalg.norm(np.array(max_coords) - np.array(min_coords)) / 2.0
        sphere_center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]
        ax.scatter(*sphere_center, c='red', s=5)
        
        for child in node.children:
            plot_octree(child, ax)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='blue', alpha=0.5)

plot_octree(octree_root, ax)

ax.set_xlim(min_coords[0], max_coords[0])
ax.set_ylim(min_coords[1], max_coords[1])
ax.set_zlim(min_coords[2], max_coords[2])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='blue', alpha=0.5)
plot_octree(octree_root, ax)
ax.set_xlim(min_coords[0], max_coords[0])
ax.set_ylim(min_coords[1], max_coords[1])
ax.set_zlim(min_coords[2], max_coords[2])
plt.show()