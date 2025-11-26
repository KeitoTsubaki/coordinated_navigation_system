//
// Created by lihao on 19-7-9.
//

/* Modified for VW-Astar

- Node: add formation index, update constructor
- cmp: 
- VWAstarConfig: add InflateRadiusList instead of InflateRadius, FormationCosts
- VWAstar class: update point2index, index2point is commented due to complexity

*/ 

#ifndef VWASTAR_H
#define VWASTAR_H

#include <iostream>
#include <queue>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


namespace pathplanning{

enum NodeType{
    obstacle = 0,
    free,
    inOpenList,
    inCloseList
};

struct Node{
    Point point;  // node coordinate
    int formation_idx;
    double F, G, H;  // cost
    Node* parent; // parent node

    Node(Point _point = Point(0, 0), int _fmt_idx = 0):point(_point), formation_idx(_fmt_idx), F(0.0), G(0.0), H(0.0), parent(NULL)
    {
    }
};

struct cmp
{
    bool operator() (pair<double, Node*> a, pair<double, Node*> b) // Comparison function for priority queue
    {
        return a.first > b.first; // min heap
    }
};


struct VWAstarConfig{
    bool Euclidean;         // true/false
    int OccupyThresh;       // 0~255
    // int InflateRadius;      // integer

    std::vector<double> InflateRadiusList;
    std::vector<double> FormationCosts;

    VWAstarConfig(bool _Euclidean = true, int _OccupyThresh = -1):
        Euclidean(_Euclidean), OccupyThresh(_OccupyThresh)
    {
    }
};

class VWAstar{

public:
    // Interface function
    void InitVWAstar(Mat& _Map, VWAstarConfig _config = VWAstarConfig());
    // void InitVWAstar(Mat& _Map, Mat& Mask, VWAstarConfig _config = VWAstarConfig());
    void InitVWAstar(Mat& _Map, vector<Mat>& Masks, VWAstarConfig _config = VWAstarConfig());
    void PathPlanning(Point _startPoint, Point _targetPoint, vector<Point>& path);
    void DrawPath(Mat& _Map, vector<Point>& path, InputArray Mask = noArray(), Scalar color = Scalar(0, 0, 255),
            int thickness = 1, Scalar maskcolor = Scalar(255, 255, 255));

    inline int point2index(Point point, int formation_idx) {
        return (point.y * Map.cols + point.x) * config.InflateRadiusList.size() + formation_idx;
    }
    // inline Point index2point(int index) {
    //     return Point(int(index / Map.cols), index % Map.cols);
    // }

private:
    // void MapProcess(Mat& Mask);
    void MapProcess(vector<Mat>& Masks);
    Node* FindPath();
    void GetPath(Node* TailNode, vector<Point>& path);

    vector<Point> SmoothPath(const vector<Node*>& nodes);
    bool CheckLineCollision(Point p1, Point p2, int formation_idx);

private:
    //Object
    Mat Map;
    Point startPoint, targetPoint;
    Mat neighbor;

    // Mat LabelMap;
    vector<Mat> LabelMapList;
    VWAstarConfig config;

    priority_queue<pair<double, Node*>, vector<pair<double, Node*>>, cmp> OpenList; // open list
    unordered_map<int, Node*> OpenDict; // open dict
    vector<Node*> PathList;  // path list

    vector<Node*> allocated_nodes; // store all allocated nodes to free memory
};

}




#endif //VWASTAR_H