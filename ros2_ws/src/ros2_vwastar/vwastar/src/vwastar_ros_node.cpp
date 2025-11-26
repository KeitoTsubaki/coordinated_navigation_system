/**
 * @file vwastar_ros_node.cpp
 * @author Pablo Gutiérrez (pablo.gutierrez@htwg-konstanz.de)
 * @brief 
 * @version 0.1
 * @date 2024-10-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "VWAstar/vwastar_ros_node.hpp"

using namespace pathplanning;

VWAstarROSNode::VWAstarROSNode():
Node("vwastar_node")
{

    this->declare_parameter<bool>("Euclidean", true);
    this->declare_parameter<bool>("ros_service", true);
    this->declare_parameter<int>("OccupyThresh", -1);
    this->declare_parameter<int>("rate", -1);
    this->declare_parameter<std::vector<double>>("InflateRadiusList", std::vector<double>{1.5, 0.5});
    this->declare_parameter<std::vector<double>>("FormationCosts", std::vector<double>{0.0, 1.0});

    config.Euclidean = this->get_parameter("Euclidean").as_bool();
    config.OccupyThresh = this->get_parameter("OccupyThresh").as_int();
    config.InflateRadiusList = this->get_parameter("InflateRadiusList").as_double_array();
    config.FormationCosts = this->get_parameter("FormationCosts").as_double_array();
    rate = this->get_parameter("rate").as_int();
    ros_service = this->get_parameter("ros_service").as_bool();

    //Services
    if(ros_service)
    {
        service_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        get_path_srv = this->create_service<path_msgs::srv::GetPath>(
                                            "/vwastar/get_path", 
                                            std::bind(&VWAstarROSNode::getPathCallback, this, _1, _2),
                                            rmw_qos_profile_services_default,
                                            service_cb_group_);

        startPoint_pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 10);
        targetPoint_pub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);
    }
    else
    {
        // Receive the start point
        startPoint_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose",
            10,
            std::bind(&VWAstarROSNode::StartPointCallback, this, _1));

        // Receive the target point
        targetPoint_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose",
            10,
            std::bind(&VWAstarROSNode::TargetPointCallback, this, _1));
        //Timers
        timer_ = this->create_wall_timer(std::chrono::seconds(1/rate),std::bind(&VWAstarROSNode::timer_callback, this));

    }

    //Subscribe map
    map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 
        10, 
        std::bind(&VWAstarROSNode::MapCallback, this, _1));


    //Publishers
    mask_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/vwastar/mask", 10);
    path_pub = this->create_publisher<nav_msgs::msg::Path>("/vwastar/nav_path", 10);
    robot1_formation_target_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/vwastar/robot1_formation_target", 10);
    robot2_formation_target_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/vwastar/robot2_formation_target", 10);

    OccGridParam = std::make_unique<OccupancyGridParam>();
    vwastar = std::make_unique<pathplanning::VWAstar>();

    RCLCPP_INFO(this->get_logger(), "VWAstar node successfully started!");

}

geometry_msgs::msg::Quaternion VWAstarROSNode::createQuaternionFromYaw(double yaw)
{
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = sin(yaw * 0.5);
    q.w = cos(yaw * 0.5);

    return q;
}

void VWAstarROSNode::getPathCallback(const std::shared_ptr<path_msgs::srv::GetPath::Request> request,
                                   std::shared_ptr<path_msgs::srv::GetPath::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Getting path...");

    //response->success = true;
    Point2d start_point = Point2d(request->init_pose.pose.position.x, request->init_pose.pose.position.y);
    Point2d goal_point = Point2d(request->goal_pose.pose.position.x, request->goal_pose.pose.position.y);
    OccGridParam->Map2ImageTransform(start_point, startPoint);
    OccGridParam->Map2ImageTransform(goal_point, targetPoint);

    geometry_msgs::msg::PoseWithCovarianceStamped startPointMsg;
    startPointMsg.header.stamp = Node::now();
    startPointMsg.header.frame_id = "map";
    startPointMsg.pose.pose.position.x = request->init_pose.pose.position.x;
    startPointMsg.pose.pose.position.y = request->init_pose.pose.position.y;
    startPointMsg.pose.pose.position.z = 0;
    startPointMsg.pose.pose.orientation = request->init_pose.pose.orientation;

    startPoint_pub->publish(startPointMsg);

    geometry_msgs::msg::PoseStamped goalPointMsg;
    goalPointMsg.header.stamp = Node::now();
    goalPointMsg.header.frame_id = "map";
    goalPointMsg.pose.position.x = request->goal_pose.pose.position.x;
    goalPointMsg.pose.position.y = request->goal_pose.pose.position.y;
    goalPointMsg.pose.position.z = 0;
    goalPointMsg.pose.orientation = request->goal_pose.pose.orientation;

    targetPoint_pub->publish(goalPointMsg);

    start_orientation = request->init_pose.pose.orientation;
    target_orientation = request->goal_pose.pose.orientation;
    
    startpoint_flag = true;
    targetpoint_flag = true;

    if(map_flag)
    {
        start_flag = true;
        this->compute_path();
    }

    response->path = this->path;

}


void VWAstarROSNode::MapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Map Received!");

    // Get parameter
    OccGridParam->GetOccupancyGridParam(msg);

    // Get map
    int height = OccGridParam->height;
    int width = OccGridParam->width;
    int OccProb;
    Mat Map(height, width, CV_8UC1);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            OccProb = msg->data[i * width + j];
            OccProb = (OccProb < 0) ? 100 : OccProb; // set Unknown to 0
            // The origin of the OccGrid is on the bottom left corner of the map
            Map.at<uchar>(height-i-1, j) = 255 - round(OccProb * 255.0 / 100.0);
        }
    }

    // Initial VWAstar
    // Mat Mask;
    vector<Mat> Masks;

    std::vector<double> InflateRadiusList = this->get_parameter("InflateRadiusList").as_double_array();
    config.InflateRadiusList.clear();
    for(const auto& radius : InflateRadiusList)
    {
        config.InflateRadiusList.push_back(round(radius / OccGridParam->resolution));
    }

    config.FormationCosts = this->get_parameter("FormationCosts").as_double_array();

    // vwastar->InitVWAstar(Map, Mask, config);
    vwastar->InitVWAstar(Map, Masks, config);

    // Publish Mask
    OccGridMask.header.stamp = Node::now();
    OccGridMask.header.frame_id = "map";
    OccGridMask.info = msg->info;
    OccGridMask.data.clear();

    int num_masks = Masks.size();
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            int prob = 0;
            for(int m=0; m<num_masks; m++)
            {
                prob += Masks[m].at<uchar>(height-i-1, j) * 50;
            }
            OccGridMask.data.push_back(prob);
        }
    }
    // for(int i=0;i<height;i++)
    // {
    //     for(int j=0;j<width;j++)
    //     {
    //         OccProb = Mask.at<uchar>(height-i-1, j) * 175;
    //         OccGridMask.data.push_back(OccProb);
    //     }
    // }

    // Set flag
    map_flag = true;
    startpoint_flag = false;
    targetpoint_flag = false;   
}

void VWAstarROSNode::StartPointCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Start Point Received!");

    Point2d src_point = Point2d(msg->pose.pose.position.x, msg->pose.pose.position.y);
    OccGridParam->Map2ImageTransform(src_point, startPoint);

    start_orientation = msg->pose.pose.orientation;

    // Set flag
    startpoint_flag = true;
    if(map_flag && startpoint_flag && targetpoint_flag)
    {
        start_flag = true;
    }

}


void VWAstarROSNode::TargetPointCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Target Point Received!");

    RCLCPP_INFO(this->get_logger(), "map_flag: %d, startpoint_flag: %d, targetpoint_flag: %d", map_flag, startpoint_flag, targetpoint_flag);
    Point2d src_point = Point2d(msg->pose.position.x, msg->pose.position.y);
    OccGridParam->Map2ImageTransform(src_point, targetPoint);

    target_orientation = msg->pose.orientation;

    // Set flag
    targetpoint_flag = true;
    if(map_flag && startpoint_flag && targetpoint_flag)
    {
        start_flag = true;
    }
}

void VWAstarROSNode::compute_path()
{
    if(start_flag)
    {
        std::chrono::high_resolution_clock::time_point current_time = std::chrono::high_resolution_clock::now();
        // start planning path
        std::vector<Point> PathList;
        vwastar->PathPlanning(startPoint, targetPoint, PathList);
        path.poses.clear();
        RCLCPP_INFO(this->get_logger(), "num of path points: %ld ", PathList.size());
        geometry_msgs::msg::PoseArray robot1_msg, robot2_msg;
        if(!PathList.empty())
        {
            path.header.stamp = Node::now();
            path.header.frame_id = "map";

            // まず全ての点をワールド座標に変換してリスト化
            std::vector<Point2d> world_points;
            for(auto pt : PathList) {
                Point2d dst_point;
                OccGridParam->Image2MapTransform(pt, dst_point);
                world_points.push_back(dst_point);
            }
            // Point2d dst_point;
            // OccGridParam->Image2MapTransform(PathList[i], dst_point);

            robot1_msg.header = path.header;
            robot2_msg.header = path.header;

            double offset_dist_m = 1.0;

            for (size_t i = 0; i < world_points.size(); i++)
            {
                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.stamp = Node::now();
                pose_stamped.header.frame_id = "map";

                // Position
                pose_stamped.pose.position.x = world_points[i].x;
                pose_stamped.pose.position.y = world_points[i].y;
                pose_stamped.pose.position.z = 0;
                
                // Orientation
                double yaw = 0.0;
                if(i < world_points.size() - 1)
                {
                    double dy = world_points[i+1].y - world_points[i].y;
                    double dx = world_points[i+1].x - world_points[i].x;
                    yaw = atan2(dy, dx);

                    pose_stamped.pose.orientation = createQuaternionFromYaw(yaw);
                }
                else
                {
                    double dy = world_points[i].y - world_points[i-1].y;
                    double dx = world_points[i].x - world_points[i-1].x;
                    yaw = atan2(dy, dx);

                    pose_stamped.pose.orientation = createQuaternionFromYaw(yaw);
                    // pose_stamped.pose.orientation = target_orientation;

                    // transform orientation to yaw
                    // tf2::Quaternion q(target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w);
                    // double roll, pitch;
                    // tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
                }

                path.poses.push_back(pose_stamped);

                // Make formation targets
                // x: forward, y: left
                if(i > 0)
                {
                    geometry_msgs::msg::Pose robot1_pose = pose_stamped.pose; // copy orientation
                    robot1_pose.position.x = world_points[i].x - offset_dist_m * sin(yaw);
                    robot1_pose.position.y = world_points[i].y + offset_dist_m * cos(yaw);
                    robot1_msg.poses.push_back(robot1_pose);
    
                    geometry_msgs::msg::Pose robot2_pose = pose_stamped.pose; // copy orientation
                    robot2_pose.position.x = world_points[i].x + offset_dist_m * sin(yaw);
                    robot2_pose.position.y = world_points[i].y - offset_dist_m * cos(yaw);
                    robot2_msg.poses.push_back(robot2_pose);
                }

            }
            std::chrono::high_resolution_clock::time_point last_time = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(last_time - current_time).count();
            RCLCPP_INFO(this->get_logger(), "Find a valid path successfully! Use %f [ms]", dt);

        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Can not find a valid path");
        }

        // Publish map
        path_pub->publish(path);
        robot1_formation_target_pub->publish(robot1_msg);
        robot2_formation_target_pub->publish(robot2_msg);

        // Set flag
        start_flag = false;

        if(map_flag)
        {
            // Publish occupancy grid mask
            mask_pub->publish(OccGridMask);
            
        }


    }

}


void VWAstarROSNode::timer_callback()
{
    this->compute_path();
}


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<pathplanning::VWAstarROSNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}