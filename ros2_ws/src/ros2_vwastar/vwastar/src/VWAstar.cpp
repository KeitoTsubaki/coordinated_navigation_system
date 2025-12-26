//
// Created by lihao on 19-7-9.
//

/* Modified for VW-Astar
- MapProcess: to create multiple LabelMaps according to InflateRadiusList
- FindPath: 
*/

#include "VWAstar/VWAstar.h"

namespace pathplanning{

void VWAstar::InitVWAstar(Mat& _Map, VWAstarConfig _config)
{
    // Mat Mask;
    // InitVWAstar(_Map, Mask, _config);
    vector<Mat> Masks;
    InitVWAstar(_Map, Masks, _config);
}

// void VWAstar::InitVWAstar(Mat& _Map, Mat& Mask, VWAstarConfig _config)
void VWAstar::InitVWAstar(Mat& _Map, vector<Mat>& Masks, VWAstarConfig _config)
{
    char neighbor8[8][2] = {
            {-1, -1}, {-1, 0}, {-1, 1},
            {0, -1},            {0, 1},
            {1, -1},   {1, 0},  {1, 1}
    };

    Map = _Map;
    config = _config;
    neighbor = Mat(8, 2, CV_8S, neighbor8).clone();

    // MapProcess(Mask);
    MapProcess(Masks);
}

void VWAstar::PathPlanning(Point _startPoint, Point _targetPoint, vector<PathPoint>& path)
{
    // Get variables
    startPoint = _startPoint;
    targetPoint = _targetPoint;

    // Path Planning
    Node* TailNode = FindPath();
    GetPath(TailNode, path);
}

void VWAstar::DrawPath(Mat& _Map, vector<Point>& path, InputArray Mask, Scalar color,
        int thickness, Scalar maskcolor)
{
    if(path.empty())
    {
        cout << "Path is empty!" << endl;
        return;
    }
    _Map.setTo(maskcolor, Mask);
    for(auto it:path)
    {
        rectangle(_Map, it, it, color, thickness);
    }
}

// void VWAstar::MapProcess(Mat& Mask)
void VWAstar::MapProcess(vector<Mat>& Masks)
{
    int width = Map.cols;
    int height = Map.rows;
    Mat _Map = Map.clone();

    // Transform RGB to gray image
    if(_Map.channels() == 3)
    {
        cvtColor(_Map.clone(), _Map, cv::COLOR_BGR2GRAY);
    }

    // Binarize
    if(config.OccupyThresh < 0)
    {
        threshold(_Map.clone(), _Map, 0, 255, cv::THRESH_OTSU);
    } else
    {
        threshold(_Map.clone(), _Map, config.OccupyThresh, 255, cv::THRESH_BINARY);
    }

    // ===== Make several maps for VW-Astar =====
    LabelMapList.clear();
    Masks.clear();
    int num_formations = config.InflateRadiusList.size();

    Mat CombinedSrc = _Map.clone();

    for(int i=0; i < num_formations; ++i)
    {
        // Mat src = _Map.clone();
        double radius = config.InflateRadiusList[i];
        int radius_int = static_cast<int>(std::round(radius));

        // Inflate
        if(radius > 0)
        {
            Mat se = getStructuringElement(MORPH_ELLIPSE, Size(2 * radius_int, 2 * radius_int));
            erode(CombinedSrc, _Map, se);
        }

        
        // Create LabelMap
        Mat currentLabelMap = Mat::zeros(height, width, CV_8UC1);
        for(int y=0;y<height;y++)
        {
            for(int x=0;x<width;x++)
            {
                // Seeing inflated _Map
                if(_Map.at<uchar>(y, x) == 0)
                {
                    currentLabelMap.at<uchar>(y, x) = obstacle;
                }
                else
                {
                    currentLabelMap.at<uchar>(y, x) = free;
                }
            }
        }
        LabelMapList.push_back(currentLabelMap);
        
        // Get mask
        // bitwise_xor(CombinedSrc, _Map, Mask);
        Mat Mask;
        bitwise_xor(CombinedSrc, _Map, Mask);
        Masks.push_back(Mask);
    }


}

Node* VWAstar::FindPath()
{

    allocated_nodes.clear();

    int width = Map.cols;
    int height = Map.rows;
    int num_formations = config.InflateRadiusList.size();

    int start_fmt = 0; // Default formation index for start point
    
    // Mat _LabelMap = LabelMap.clone();
    vector<Mat> _LabelMaps;
    for(const auto& map : LabelMapList)
    {
        _LabelMaps.push_back(map.clone());
    }
    
    // Add startPoint to OpenList
    Node* startPointNode = new Node(startPoint, start_fmt);
    allocated_nodes.push_back(startPointNode);

    startPointNode->G = 0.0;
    startPointNode->H = 0.0; 
    startPointNode->F = 0.0;
    
    // OpenList.push(pair<int, Point>(startPointNode->F, startPointNode->point));
    OpenList.push(make_pair(startPointNode->F, startPointNode));
    int index = point2index(startPointNode->point, start_fmt);

    OpenDict[index] = startPointNode;

    _LabelMaps[start_fmt].at<uchar>(startPoint.y, startPoint.x) = inOpenList;

    while(!OpenList.empty())
    {
        // Find the node with least F value
        Node* CurNode = OpenList.top().second;
        OpenList.pop();

        Point CurPoint = CurNode->point;
        int cur_fmt = CurNode->formation_idx;

        int index = point2index(CurPoint, cur_fmt);

        if (OpenDict.find(index) == OpenDict.end()) {
            continue; // This node has been processed
        }
        OpenDict.erase(index);

        int curX = CurPoint.x;
        int curY = CurPoint.y;
        _LabelMaps[cur_fmt].at<uchar>(curY, curX) = inCloseList;

        // Determine whether arrive the target point
        if(curX == targetPoint.x && curY == targetPoint.y)
        {
            return CurNode; // Find a valid path
        }

        // Traversal the neighborhood
        for(int k = 0;k < neighbor.rows;k++)
        {
            int y = curY + neighbor.at<char>(k, 0);
            int x = curX + neighbor.at<char>(k, 1);
            if(x < 0 || x >= width || y < 0 || y >= height)
            {
                continue;
            }

            for(int next_fmt = 0; next_fmt < num_formations; next_fmt++)
            {
                uchar status = _LabelMaps[next_fmt].at<uchar>(y, x);
                
                if(status == obstacle || status == inCloseList)
                {
                    continue;
                }

                // Determine whether a diagonal line can pass
                int dist1 = abs(neighbor.at<char>(k, 0)) + abs(neighbor.at<char>(k, 1));
                if(dist1 == 2 && _LabelMaps[next_fmt].at<uchar>(y, curX) == obstacle && _LabelMaps[next_fmt].at<uchar>(curY, x) == obstacle)
                    continue;
    
                // Calculate G, H, F value
                double addG, G, H, F;
                if(dist1 == 2)
                {
                    addG = 14.0;
                }
                else
                {
                    addG = 10.0;
                }

                double fmt_cost = config.FormationCosts[next_fmt];

                G = CurNode->G + addG + fmt_cost;

                if(config.Euclidean)
                {
                    double dist2 = (x - targetPoint.x) * (x - targetPoint.x) + (y - targetPoint.y) * (y - targetPoint.y);
                    H = round(10 * sqrt(dist2));
                }
                else
                {
                    H = 10 * (abs(x - targetPoint.x) + abs(y - targetPoint.y));
                }
                F = G + H;
    
                // Update the G, H, F value of node
                if(status == free)
                {
                    Node* node = new Node(Point(x, y), next_fmt);
                    allocated_nodes.push_back(node);

                    node->parent = CurNode;
                    node->G = G;
                    node->H = H;
                    node->F = F;

                    OpenList.push(make_pair(node->F, node));
                    int index = point2index(node->point, next_fmt);
                    OpenDict[index] = node;

                    _LabelMaps[next_fmt].at<uchar>(y, x) = inOpenList;
                }
                else // status == inOpenList
                {
                    // Find the node
                    int index = point2index(Point(x, y), next_fmt);
                    if(OpenDict.find(index) != OpenDict.end()){
                        Node* node = OpenDict[index];
                        if(G < node->G)
                        {
                            node->G = G;
                            node->F = F;
                            node->parent = CurNode;
                            
                            OpenList.push(make_pair(node->F, node));
                        }
                    }
                }
            }
        }
    }

    return NULL; // Can not find a valid path
}

bool VWAstar::CheckLineCollision(Point p1, Point p2, int formation_idx) {
    // OpenCVのLineIteratorを使って2点間の全ピクセルを走査
    LineIterator it(Map, p1, p2, 8);
    for(int i = 0; i < it.count; i++, ++it) {
        Point p = it.pos();
        // マップ範囲外チェック
        if (p.x < 0 || p.x >= Map.cols || p.y < 0 || p.y >= Map.rows) return true;
        
        // 指定された形態(formation_idx)のマップで障害物があるか確認
        // LabelMapList[formation_idx] は 0=Obstacle, 255=Free の前提
        if (LabelMapList[formation_idx].at<uchar>(p.y, p.x) == obstacle) {
            return true; // 衝突あり
        }
    }
    return false; // 衝突なし(通れる)
}

vector<PathPoint> VWAstar::SmoothPath(const vector<Node*>& nodes) {
    if (nodes.empty()) return {};
    
    vector<PathPoint> smoothed_path;
    vector<Node*> waypoints;
    
    // スタート地点を登録
    waypoints.push_back(nodes[0]);
    
    int current_idx = 0;
    
    // ゴールに到達するまでループ
    while(current_idx < (int)nodes.size() - 1) {
        int valid_idx = current_idx + 1;
        int current_fmt = nodes[current_idx]->formation_idx;
        Point current_pt = nodes[current_idx]->point;
        
        // current_idx + 1 から順に見ていき、直線を引ける限界を探す
        for(int i = current_idx + 1; i < (int)nodes.size(); ++i) {
            
            // ▼▼▼ 追加: フォーメーションIDが変化したら、そこで直線を打ち切る ▼▼▼
            if (nodes[i]->formation_idx != current_fmt) {
                // 変化点の直前(i-1)までを採用する。
                // ただし、いきなり隣(current_idx+1)で変化している場合はそこを採用。
                if (i == current_idx + 1) {
                    valid_idx = i;
                } else {
                    valid_idx = i - 1;
                }
                break;
            }
            // ▲▲▲ 追加ここまで ▲▲▲
            // 現在の地点(current)から候補地点(nodes[i])への直線チェック
            // ※形態は「出発点(current)」のものを使用して判定します
            bool is_collision = CheckLineCollision(current_pt, nodes[i]->point, current_fmt);

            if(is_collision) {
                // 衝突した場合、その一つ手前(i-1)までが有効だった直線の限界
                // ただし、iがcurrentのすぐ隣(current+1)で衝突した場合は
                // 元のパスが壁ギリギリか斜め移動不可判定などをすり抜けている可能性があるため
                // 強制的に隣のノードへ進む (Stuck防止)
                if (i == current_idx + 1) {
                    valid_idx = i;
                } else {
                    valid_idx = i - 1;
                }
                break;
            }
            // 衝突しなければ、そこまでは直線でいける
            valid_idx = i;
        }
        
        // 求まったポイントを次の起点として登録
        waypoints.push_back(nodes[valid_idx]);
        current_idx = valid_idx;
    }
    // for(size_t i = 0; i < waypoints.size(); i++) {
    //     Point pos = waypoints[i]->point;
    //     smoothed_path.push_back(pos);
    // }
    
    // ウェイポイント間を直線補間してパス座標(smoothed_path)を生成
    for(size_t i = 0; i < waypoints.size() - 1; ++i) {
        Point p1 = waypoints[i]->point;
        Point p2 = waypoints[i+1]->point;
        int fmt_idx = waypoints[i]->formation_idx; // この区間のフォーメーション
        
        LineIterator it(Map, p1, p2, 8);
        for(int k = 0; k < it.count; k++, ++it) {
            if(i > 0 && k == 0) continue; 
            
            // 座標とフォーメーションIDをセットで保存
            smoothed_path.push_back({it.pos(), fmt_idx});
        }
    }

    // vector<PathPoint> spaced_smoothed_path;
    // int step = 20;
    // for(size_t i=0; i<smoothed_path.size(); i+=step)
    // {
    //     spaced_smoothed_path.push_back(smoothed_path[i]);
    // }
    // // 最後の点を確実に追加
    // if (spaced_smoothed_path.empty() || spaced_smoothed_path.back().point != smoothed_path.back().point)
    // {
    //     spaced_smoothed_path.push_back(smoothed_path.back());
    // }
    
    // return spaced_smoothed_path;

    // ▼▼▼ 修正: 間引き処理を削除し、全ての点を返す ▼▼▼
    return smoothed_path;
}

void VWAstar::GetPath(Node* TailNode, vector<PathPoint>& path)
{
    PathList.clear();
    path.clear();

    // Save path to PathList
    Node* CurNode = TailNode;
    while(CurNode != NULL)
    {
        PathList.push_back(CurNode);
        CurNode = CurNode->parent;
    }

    std::reverse(PathList.begin(), PathList.end());

    // SmoothPathは既に vector<PathPoint> を返すように修正済みなので、
    // ここで型が一致するようになります。
    path = SmoothPath(PathList);

    // Release memory
    for(Node* node : allocated_nodes)
    {
        delete node;
    }
    allocated_nodes.clear();
    while(!OpenList.empty()){
        OpenList.pop();
    }
    OpenDict.clear();
}

}