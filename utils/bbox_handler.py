import numpy as np
import math
import rospy
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from voxelnet.msg import bbox

class BOXHandler():
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, rz=None):
        self.lidar_points = None # TODO

        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.w = w
        self.l = l
        self.rz = rz

        self.min_x = None
        self.min_y = None
        self.min_z = None
        self.max_x = None
        self.max_y = None
        self.max_z = None

        self.rotated_min_x = None
        self.rotated_min_y = None
        self.rotated_min_z = None
        self.rotated_max_x = None
        self.rotated_max_y = None
        self.rotated_max_z = None

        # subscriber
        # self.sub = rospy.Subscriber('/ouster/points', PointCloud2, self.get_lidar, queue_size = 1)
        self.sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.get_lidar, queue_size = 1)
        
        # publisher
        self.pub = rospy.Publisher('/detector', bbox, queue_size=10)

    def get_lidar(self, msg):
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)
        points = np.array(list(points))
        self.lidar_points = points[:,0:4] # x,y,z,intensity

    def setParameter(self,bounding_box_info):
        self.x = bounding_box_info[0]
        self.y = bounding_box_info[1]
        self.z = bounding_box_info[2]
        self.h = bounding_box_info[3]
        self.w = bounding_box_info[4]
        self.l = bounding_box_info[5]
        self.rz = bounding_box_info[6]

    def calculate_bbox_coordinate(self):
        self.min_x = self.x - (self.l/2.0)
        self.max_x = self.x + (self.l/2.0)
        self.min_y = self.y - (self.w/2.0)
        self.max_y = self.y + (self.w/2.0)
        self.min_z = self.z - (self.h/2.0)
        self.max_z = self.z + (self.h/2.0)

    def bbox_publisher(self, bounding_box_info):
        box = bbox()        
        num_of_bbox = len(bounding_box_info)
        bounding_box_info = bounding_box_info.astype('float32') # convert type

        for i in range(num_of_bbox):
            self.setParameter(bounding_box_info[i])
            self.calculate_bbox_coordinate() # get coordinate of bounding box

            box.x_min.append(self.min_x)
            box.y_min.append(self.min_y)
            box.z_min.append(self.min_z)
            box.x_max.append(self.max_x)
            box.y_max.append(self.max_y)
            box.z_max.append(self.max_z)

        self.pub.publish(box)

    def test(self):
        rospy.init_node('velodyne_subscriber', anonymous=True)
        rospy.Subscriber('/velodyne_points', PointCloud2, self.get_lidar) 