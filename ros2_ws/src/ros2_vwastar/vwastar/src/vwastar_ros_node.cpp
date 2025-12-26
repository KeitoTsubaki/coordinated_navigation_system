/**
 * @file vwastar_ros_node.cpp
 * @author Pablo Gutiérrez (pablo.gutierrez@htwg-konstanz.de)
 * @brief 
 * @version 0.1
 * @date 2024-10-11
 * * @copyright Copyright (c) 2024
 * */

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include "VWAstar/vwastar_ros_node.hpp"
#include <rclcpp/qos.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

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
    this->declare_parameter<double>("formation_width", 1.0);

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
            "initialpose",
            10,
            std::bind(&VWAstarROSNode::StartPointCallback, this, _1));

        // Receive the target point
        targetPoint_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose",
            10,
            std::bind(&VWAstarROSNode::TargetPointCallback, this, _1));
        //Timers
        timer_ = this->create_wall_timer(std::chrono::seconds(1/rate),std::bind(&VWAstarROSNode::timer_callback, this));

    }

    rclcpp::QoS pose_qos(10);
    pose_qos.reliable(); 

    robot1_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/whill1/pose", pose_qos, 
        std::bind(&VWAstarROSNode::robot1PoseCallback, this, std::placeholders::_1));
        
    robot2_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/whill2/pose", pose_qos, 
        std::bind(&VWAstarROSNode::robot2PoseCallback, this, std::placeholders::_1));

    //Subscribe map
    // 【修正箇所】QoS設定を変更
    rclcpp::QoS map_qos(10);
    map_qos.transient_local();
    map_qos.reliable();
    map_qos.keep_last(1);

    map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "map", 
        map_qos, 
        std::bind(&VWAstarROSNode::MapCallback, this, _1));


    //Publishers
    mask_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/vwastar/mask", 10);
    path_pub = this->create_publisher<nav_msgs::msg::Path>("/vwastar/nav_path", 10);
    plan_marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("/vwastar/plan_markers", 10);
    robot1_formation_target_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/vwastar/robot1_formation_target", 10);
    robot2_formation_target_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/vwastar/robot2_formation_target", 10);

    OccGridParam = std::make_unique<OccupancyGridParam>();
    vwastar = std::make_unique<pathplanning::VWAstar>();

    RCLCPP_INFO(this->get_logger(), "VWAstar node successfully started!");

}

void VWAstarROSNode::robot1PoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    robot1_pose_ = msg->pose.pose; // PoseWithCovarianceStamped は pose.pose に Pose が入っています
    robot1_received_ = true;
}

void VWAstarROSNode::robot2PoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    robot2_pose_ = msg->pose.pose;
    robot2_received_ = true;
}

// ... 以降のコードは変更なし ...
// (createQuaternionFromYaw, getPathCallback, MapCallback などはそのまま)
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
    // Point2d start_point = Point2d(request->init_pose.pose.position.x, request->init_pose.pose.position.y);
    // Point2d goal_point = Point2d(request->goal_pose.pose.position.x, request->goal_pose.pose.position.y);
    // ▼▼▼ 変更: 中間地点の計算とスタート地点の上書き ▼▼▼
    double start_x, start_y;
    geometry_msgs::msg::Quaternion start_quat;

    if (robot1_received_ && robot2_received_) {
        // 2台の中間点を計算
        start_x = (robot1_pose_.position.x + robot2_pose_.position.x) / 2.0;
        start_y = (robot1_pose_.position.y + robot2_pose_.position.y) / 2.0;
        
        // 向きはリクエスト（現在のリーダーまたは指定）を使うか、2台の平均を使う
        // ここではリクエストの向き（ゴールへ向かう意志）を優先しますが、必要なら平均化してください
        start_quat = request->init_pose.pose.orientation;
        
        RCLCPP_INFO(this->get_logger(), "Planning from Group Center: (%f, %f)", start_x, start_y);
    } else {
        // ロボットの位置が未受信ならリクエストの値をそのまま使う（フォールバック）
        start_x = request->init_pose.pose.position.x;
        start_y = request->init_pose.pose.position.y;
        start_quat = request->init_pose.pose.orientation;
        RCLCPP_WARN(this->get_logger(), "Group center unknown, using request pose.");
    }

    Point2d start_point = Point2d(start_x, start_y);
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
        std::vector<PathPoint> PathList;
        vwastar->PathPlanning(startPoint, targetPoint, PathList);
        path.poses.clear();
        RCLCPP_INFO(this->get_logger(), "num of path points: %ld ", PathList.size());
        geometry_msgs::msg::PoseArray robot1_msg, robot2_msg;
        if(!PathList.empty())
        {
            path.header.stamp = Node::now();
            path.header.frame_id = "map";

            // まず全ての点をワールド座標に変換してリスト化
            struct WorldPathPoint {
                Point2d pos;
                int fmt_idx;
            };
            std::vector<WorldPathPoint> world_points;
            for(auto pt : PathList) {
                Point2d dst_point;
                OccGridParam->Image2MapTransform(pt.point, dst_point); // pt.pointにアクセス
                world_points.push_back({dst_point, pt.formation_idx}); // fmt_idxを保存
            }

            double formation_width = this->get_parameter("formation_width").as_double();
            double offset_dist_m = formation_width / 2.0;

            robot1_msg.header = path.header;
            robot2_msg.header = path.header;

            for (size_t i = 0; i < world_points.size(); i++)
            {
                geometry_msgs::msg::PoseStamped pose_stamped;
                pose_stamped.header.stamp = Node::now();
                pose_stamped.header.frame_id = "map";

                // Position
                // ▼▼▼ 修正: .pos を追加 ▼▼▼
                pose_stamped.pose.position.x = world_points[i].pos.x;
                pose_stamped.pose.position.y = world_points[i].pos.y;
                pose_stamped.pose.position.z = static_cast<double>(world_points[i].fmt_idx);
                
                // Orientation
                double yaw = 0.0;
                int tangent_lookahead_steps = 3;
                if(i < world_points.size() - tangent_lookahead_steps)
                {
                    // ▼▼▼ 修正: .pos を追加 ▼▼▼
                    double dy = world_points[i+tangent_lookahead_steps].pos.y - world_points[i].pos.y;
                    double dx = world_points[i+tangent_lookahead_steps].pos.x - world_points[i].pos.x;
                    yaw = atan2(dy, dx);

                    pose_stamped.pose.orientation = createQuaternionFromYaw(yaw);
                }
                else
                {
                    // ▼▼▼ 修正: .pos を追加 ▼▼▼
                    double dy = world_points[i].pos.y - world_points[i-1].pos.y;
                    double dx = world_points[i].pos.x - world_points[i-1].pos.x;
                    yaw = atan2(dy, dx);

                    pose_stamped.pose.orientation = createQuaternionFromYaw(yaw);
                }

                path.poses.push_back(pose_stamped);

                // Make formation targets
                // x: forward, y: left
                // ▼▼▼ フォーメーションごとのターゲット生成分岐 ▼▼▼
                if(i > 0) 
                {
                    geometry_msgs::msg::Pose robot1_pose = pose_stamped.pose; 
                    geometry_msgs::msg::Pose robot2_pose = pose_stamped.pose;
                    
                    int current_fmt = world_points[i].fmt_idx;

                    // 仮定: InflateRadiusList[0] が広い(Line), [1] が狭い(Column)
                    if (current_fmt == 0) 
                    {
                        // === 横並び (Line Formation) ===
                        // 既存のロジック: パスに対して左右に展開
                        robot1_pose.position.x = world_points[i].pos.x - offset_dist_m * sin(yaw);
                        robot1_pose.position.y = world_points[i].pos.y + offset_dist_m * cos(yaw);
                        
                        robot2_pose.position.x = world_points[i].pos.x + offset_dist_m * sin(yaw);
                        robot2_pose.position.y = world_points[i].pos.y - offset_dist_m * cos(yaw);
                    }
                    else 
                    {
                        // === 縦並び (Column Formation) ===
                        // 狭いので左右に広げず、センターライン上（または極小オフセット）にする
                        // ここではシンプルに「2台ともパスの中心を狙う」設定にします
                        // ※衝突回避はLocal Planner (RL) に任せるか、前後方向の遅延を入れる必要があります
                        
                        robot1_pose.position.x = world_points[i].pos.x;
                        robot1_pose.position.y = world_points[i].pos.y;
                        
                        robot2_pose.position.x = world_points[i].pos.x;
                        robot2_pose.position.y = world_points[i].pos.y;
                        
                        // オプション: 少しだけ前後にずらす場合
                        // robot1_pose.position.x += 0.5 * cos(yaw); ...
                    }

                    robot1_msg.poses.push_back(robot1_pose);
                    robot2_msg.poses.push_back(robot2_pose);
                }

            }

            // === ▼▼▼ ここから追加: マーカーによる可視化 ▼▼▼ ===
            visualization_msgs::msg::Marker points_marker;
            points_marker.header = path.header; // frame_id と time を合わせる
            points_marker.ns = "formation_path";
            points_marker.id = 0;
            points_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            points_marker.action = visualization_msgs::msg::Marker::ADD;
            
            // 点のサイズ (メートル)
            points_marker.scale.x = 0.1; 
            points_marker.scale.y = 0.1;
            points_marker.scale.z = 0.1;

            // 共通のアルファ値（不透明度）
            points_marker.color.a = 1.0; 

            // world_points は前回のコード修正案で使用した struct WorldPathPoint のリスト
            for(const auto& wp : world_points)
            {
                geometry_msgs::msg::Point p;
                p.x = wp.pos.x;
                p.y = wp.pos.y;
                p.z = 0.1; // 地図に埋もれないように少し浮かせる
                points_marker.points.push_back(p);

                std_msgs::msg::ColorRGBA c;
                c.a = 1.0;
                
                // 隊形IDに応じた色分け
                if(wp.fmt_idx == 0) {
                    // ID 0 (Line/Wide): 青
                    c.r = 0.0; c.g = 0.0; c.b = 1.0; 
                }
                else if (wp.fmt_idx == 1) {
                    // ID 1 (Column/Narrow): 赤
                    c.r = 1.0; c.g = 0.0; c.b = 0.0;
                }
                else {
                    // その他: 緑
                    c.r = 0.0; c.g = 1.0; c.b = 0.0;
                }
                points_marker.colors.push_back(c);
            }
            
            plan_marker_pub->publish(points_marker);
            // === ▲▲▲ 追加ここまで ▲▲▲ ===

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