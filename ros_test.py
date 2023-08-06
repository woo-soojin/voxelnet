#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import tensorflow as tf

from model import RPN3D
from config import cfg
from utils import *
from utils.bbox_handler import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                        help='set log tag')
    parser.add_argument('--output-path', type=str, nargs='?',
                        default='./predictions', help='results output dir')
    parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=2,
                        help='set batch size for each gpu')
    parser.add_argument('-v', '--vis', type=bool, nargs='?', default=False,
                        help='set the flag to True if dumping visualizations')
    args = parser.parse_args()

    dataset_dir = cfg.DATA_DIR
    val_dir = os.path.join(cfg.DATA_DIR, 'validation')
    save_model_dir = os.path.join('./save_model', args.tag)
    
    # create output folder
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'data'), exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(args.output_path, 'vis'), exist_ok=True)


    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                            visible_device_list=cfg.GPU_AVAILABLE,
                            allow_growth=True)
    
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            device_count={
                "GPU": cfg.GPU_USE_COUNT,
            },
            allow_soft_placement=True,
        )

        with tf.Session(config=config) as sess:
            model = RPN3D(
                cls=cfg.DETECT_OBJ,
                single_batch_size=args.single_batch_size,
                avail_gpus=cfg.GPU_AVAILABLE.split(',')
            )
            if tf.train.get_checkpoint_state(save_model_dir):
                print("Reading model parameters from %s" % save_model_dir)
                model.saver.restore(
                    sess, tf.train.latest_checkpoint(save_model_dir))
            
            rospy.init_node('VoxelNet', anonymous=True) # ROS
            node = BOXHandler(model, sess, args.single_batch_size, cfg.GPU_USE_COUNT)
            rospy.spin()