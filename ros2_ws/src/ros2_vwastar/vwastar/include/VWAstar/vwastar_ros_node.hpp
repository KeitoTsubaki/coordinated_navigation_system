/**
 * @file vwastar_ros_node.hpp
 * @author Pablo Gutiérrez (pablo.gutierrez@htwg-konstanz.de)
 * @brief 
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef VWASTAR_ROS_NODE_HPP_
#define VWASTAR_ROS_NODE_HPP_

#include <iostream>
#include <cmath>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include "path_msgs/srv/get_path.hpp"
#include "path_msgs/srv/pose.hpp"
//#include <nav2_msgs/srv/is_path_valid.hpp>

#include <opencv2/opencv.hpp>
#include "VWAstar.h"
#include "OccMapTransform.h"

#include <visualization_msgs/msg/marker.hpp>

using namespace cv;
using namespace std;

using namespace std::chrono_literals;
using namespace std::placeholders;

namespace pathplanning
{

class VWAstarROSNode: public rclcpp::Node
{
private:

  
  nav_msgs::msg::OccupancyGrid OccGridMask;
  nav_msgs::msg::Path path;
  pathplanning::VWAstarConfig config;
  std::unique_ptr<pathplanning::VWAstar> vwastar;
  std::unique_ptr<OccupancyGridParam> OccGridParam;
  Point startPoint, targetPoint;

  geometry_msgs::msg::Quaternion start_orientation;
  geometry_msgs::msg::Quaternion target_orientation;

  // Parameter
  double InflateRadius;
  bool map_flag;
  bool startpoint_flag;
  bool targetpoint_flag;
  bool start_flag;
  bool ros_service;
  int rate;

  std::chrono::high_resolution_clock::time_point wheel_last_time;
  std::chrono::steady_clock::time_point last_time;
  rclcpp::TimerBase::SharedPtr timer_;

  //Subscribers
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr startPoint_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoint_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr robot1_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr robot2_sub_;
  


  //Publishers
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mask_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr robot1_formation_target_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr robot2_formation_target_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr startPoint_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoint_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plan_marker_pub;

  //Services
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  rclcpp::Service<path_msgs::srv::GetPath>::SharedPtr get_path_srv;

  geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw);

  // ▼▼▼ 追加: ロボットの位置と受信フラグを保持する変数 ▼▼▼
  geometry_msgs::msg::Pose robot1_pose_;
  geometry_msgs::msg::Pose robot2_pose_;
  bool robot1_received_ = false;
  bool robot2_received_ = false;
  // ▲▲▲ 追加ここまで ▲▲▲

  // Service callbacks
  void getPathCallback(const std::shared_ptr<path_msgs::srv::GetPath::Request> request,
                       std::shared_ptr<path_msgs::srv::GetPath::Response> response);

  // Subscriber callbacks
  void MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void StartPointCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void TargetPointCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void compute_path();
  void timer_callback();



public:
  
  VWAstarROSNode();
  ~VWAstarROSNode(){}

  void robot1PoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void robot2PoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);


};

} //end pathplanning namespace

#endif  // VWASTAR_ROS_NODE_HPP_