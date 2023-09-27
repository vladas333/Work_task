import unittest
from app import create_octree

class TestCreateOctree(unittest.TestCase):

    def test_depth_zero(self):
        point_cloud = []  
        min_coords = [0, 0, 0]
        max_coords = [1, 1, 1]
        depth = 0

        result = create_octree(point_cloud, min_coords, max_coords, depth)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
