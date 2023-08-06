import numpy as np
import math
import rospy
import sensor_msgs.point_cloud2

from sensor_msgs.msg import PointCloud2
from voxelnet.msg import bbox
from utils.kitti_loader import build_input
from utils.preprocess import process_pointcloud

class BOXHandler():
    def __init__(self, model, sess, single_batch_size, GPU_USE_COUNT):        
        self.model = model
        self.sess = sess
        self.single_batch_size = single_batch_size
        self.GPU_USE_COUNT = GPU_USE_COUNT

        self.lidar_points = None

        self.x = None
        self.y = None
        self.z = None
        self.h = None
        self.w = None
        self.l = None
        self.rz = None

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
        self.sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback, queue_size=10) # TODO queue
        # self.sub = rospy.Subscriber('/ouster/points', PointCloud2, self.callback, queue_size=10) # TODO queue

        # publisher
        self.pub = rospy.Publisher('/detector', bbox, queue_size=10)
        self.rate = rospy.Rate(10)

    def callback(self, msg):
        msg = sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)
        points = np.array(list(msg))
        self.lidar_points = points[:,0:4]
        
        voxel_dict = process_pointcloud(self.lidar_points)
        batchs = self.iterate_data([voxel_dict], self.single_batch_size * self.GPU_USE_COUNT, self.GPU_USE_COUNT)
        for batch in batchs:
            results = self.model.ros_predict_step(self.sess, batch)

            for result in results:
                self.bbox_publisher(result[:, 1:8])

    def iterate_data(self, voxel, batch_size, multi_gpu_sum):
        vox_feature, vox_number, vox_coordinate = [], [], []
        single_batch_size = int(batch_size / multi_gpu_sum)
        for idx in range(multi_gpu_sum):
            _, per_vox_feature, per_vox_number, per_vox_coordinate = build_input(voxel[idx * single_batch_size:(idx + 1) * single_batch_size])
            vox_feature.append(per_vox_feature)
            vox_number.append(per_vox_number)
            vox_coordinate.append(per_vox_coordinate)

        ret = (
                np.array(vox_feature),
                np.array(vox_number),
                np.array(vox_coordinate)
                )

        yield ret

    def setParameter(self, bounding_box_info):
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
        self.rate.sleep() # TODO