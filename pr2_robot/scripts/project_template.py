#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    #create filter object
    outlier_filtered = pcl_cloud.make_statistical_outlier_filter()
    # Set number of neighboring points to analyse for a given points
    outlier_filtered.set_mean_k(5)
    # Set threshold scale factor
    x = 1.0
    # Consider any point with a mean distance than global mean (mean + x * std_dev)
    outlier_filtered.set_std_dev_mul_thresh(x)
    # Finally, call the filter function
    cloud_filtered = outlier_filtered.filter()
    filename = 'outlier_filtered.pcd'
    pcl.save(cloud_filtered,filename )
    
    
    # TODO: Voxel Grid Downsampling
    # Create a voxelGrid filter object the point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel or leaf size
    LEAF_SIZE = 0.01
    # Set the leaf size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # call the filter function to obtain the resultant downsampled point cloud
    cloud_downsampled = vox.filter()
    filename= 'voxel_downsampled.pcd'
    pcl.save(cloud_downsampled,filename)

    
    # TODO: PassThrough Filter
    # Create passthrough filter object
    passthrough = cloud_downsampled.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object for z axis
    filter_axis='z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.61
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min,axis_max)
    cloud_passed = passthrough.filter()
    # Assign axis and range to the passthrough filter object for y axis
    passthrough = cloud_passed.make_passthrough_filter()
    filter_axis='y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = +0.4
    passthrough.set_filter_limits(axis_min,axis_max )
    cloud_passed = passthrough.filter()
    filename ='pass_through_cloud_filtered.pcd' 
    pcl.save(cloud_passed, filename)
    
    
    # TODO: RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_passed.make_segmenter()
    # Set model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Set maximum distant to be considerred for fitiing
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inliers indices and model coefficients
    inliers, coefficients = seg.segment()
    
    # Extract inliers
    cloud_table = cloud_passed.extract(inliers, negative=False)
    filename = 'extracted_inliers.pcd'
    # Save pcd for table
    pcl.save(cloud_table, filename)

    # Extract outliers

    cloud_objects = cloud_passed.extract(inliers, negative=True)
    filename = 'extracted_outliers.pcd'
    # Save pcd for tabletop objects
    pcl.save(cloud_objects, filename)
 
    # TODO: Euclidean Clustering
    # Apply function to convert XYZRGB to XYZ
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    # Contruct k-d tree
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # set tolearance for diistance threshold as well as maximum cluster size
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(2800)
    # Search the k-d tree for cluster
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered cluster
    cluster_indices = ec.Extract()


    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color to each segmented object in the scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
            white_cloud[indice][1],
            white_cloud[indice][2],rgb_to_float(cluster_color[j])])
    # Create new cluster to contain all cluters each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    filename ='cluster_cloud.pcd'
    pcl.save(cluster_cloud, filename)
    # TODO: Convert PCL data to ROS messages
    ros_cloud_filtered = pcl_to_ros(cloud_passed)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_filtered)
    pcl_cluster_cloud_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_object_labels = []
    detected_objects = []
    for index, point_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outlier (cloud_objects)
        pcl_cluster = cloud_objects.extract(point_list)
        # convert cluster from pcl to ROS
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        color_hists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        normals_hists = compute_normal_histograms(normals)
        hists_feature = np.concatenate((color_hists, normals_hists))
        # Make the prediction
        # Retrieve the label for the result and add it to detection_objects_label
        prediction = clf.predict(scaler.transform(hists_feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_object_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[point_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objectss
    detected_objects_pub.publish(detected_objects)
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass
    
# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 2
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    object_param_dict = {}
    for idx in range(0,len(object_list_param)):
        object_param_dict[object_list_param[idx]['name']] = object_list_param[idx]
       
    dropbox_param_dict = {}
    for idx in range(0,len(dropbox_list_param)):
        dropbox_param_dict[dropbox_list_param[idx]['group']] = dropbox_list_param[idx]
       
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for object in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        point_arr = ros_to_pcl(object.cloud).to_array()
        centroid = np.mean(point_arr, axis=0)[:3]
        print object_param_dict[object.label]
        # Get config param for that kind of object
        object_param = object_param_dict[object.label]

        # Get config param for that kind of object
        dropbox_param = dropbox_param_dict[object_param['group']]
        
        object_name.data = str(object.label)

        # TODO: Create 'pick_pose' for the object
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])
        pick_pose.orientation.x =0.0
        pick_pose.orientation.y =0.0
        pick_pose.orientation.z =0.0
        pick_pose.orientation.w =0.0
        
        # TODO: Create 'place_pose' for the object 
        position = dropbox_param['position'] + np.random.rand(3)/10
        pick_pose.position.x = float(position[0])
        pick_pose.position.y = float(position[1])
        pick_pose.position.z = float(position[2])
        pick_pose.orientation.x =0.0
        pick_pose.orientation.y =0.0
        pick_pose.orientation.z =0.0
        pick_pose.orientation.w =0.0
        
        # TODO: Assign the arm to be used for pick_place
        arm_name.data = str(dropbox_param['name'])

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict_list = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict_list)

        # Wait for 'pick_place_routine' service to come up
       
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            #resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            #print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    yaml_filename ="output" + str(test_scene_num.data) + ".yaml" 
    send_to_yaml(yaml_filename, dict_list)
   

if __name__ == '__main__':

   # TODO: ROS node initialization
    rospy.init_node("clustering", anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1 )
    pcl_cluster_cloud_pub = rospy.Publisher("/pcl_cluster_cloud", PointCloud2, queue_size=1 )
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1 )
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1 )
    

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdowns
    while not rospy.is_shutdown():
        rospy.spin() 
        # TODO: publish ROS message
        print 'publishing started'
       
