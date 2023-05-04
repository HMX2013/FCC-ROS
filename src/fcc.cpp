#define PCL_NO_PRECOMPILE
#include "tools/utils.hpp"
#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <algorithm>
#include <tf/transform_listener.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#if (CV_MAJOR_VERSION == 3)
#include<opencv/cv.h>
#else
#include <opencv2/imgproc.hpp>
#endif

#include <opencv2/core/version.hpp>

#include <dynamic_reconfigure/server.h>
#include <fcc/fcc_Config.h>

#include "obsdet_msgs/CloudCluster.h"
#include "obsdet_msgs/CloudClusterArray.h"

#include "ground_truth.hpp"

using PointType = PointXYZILID;

std::string output_frame_;
std::string lidar_points_topic_;
std_msgs::Header _velodyne_header;

int WINDOW_H;
int WINDOW_W;

// Dynamic Parameters
float th_H, th_V;

tf::TransformListener *_transform_listener;
tf::StampedTransform *_transform;

static ros::Publisher pub_jskrviz_time_;
static std::chrono::time_point<std::chrono::system_clock> total_start_time_, total_end_time_;
static std_msgs::Float32 time_spent;
static double total_exe_time_ = 0.0;

boost::shared_ptr<groundtruth::DepthCluster<PointType>> gt_verify;

class FCC
{
 private:

  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  std::vector<float> vert_angles_;

  std::vector<int> index_v;

  cv::Mat rangeMat; // range matrix for range image
  cv::Mat labelMat; // label matrix for segmentaiton marking

  pcl::PointCloud<PointType>::Ptr laserCloudIn;
  PointType nanPoint; // fill in fullCloud at each iteration

  std_msgs::Header cloudHeader;
  pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
  pcl::PointCloud<PointType>::Ptr windowCloud;

  std::vector<double> ose_vector;
  std::vector<double> use_vector;

  std::vector<std::vector<int>> clusterIndices;
  std::vector<std::vector<int>> clusterIndices_ri;
  std::vector<std::vector<int>> gt_clusterIndices;

  std::vector<int> runtime_vector;

  dynamic_reconfigure::Server<fcc::fcc_Config> server;
  dynamic_reconfigure::Server<fcc::fcc_Config>::CallbackType f;

  pcl::PointCloud<pcl::PointXYZI>::Ptr segment_visul;

  ros::Subscriber sub_lidar_points;
  ros::Publisher pub_cloud_ground;
  ros::Publisher pub_jsk_bboxes;
  ros::Publisher pub_segment_visul_;
  ros::Publisher pub_obsdet_clusters_;
  ros::Publisher pub_test_visul_;

  int getRowIdx(PointType pt);
  int getColIdx(PointType pt);

  void sphericalProjection(const sensor_msgs::PointCloud2::ConstPtr &laserRosCloudMsg);
  void labelComponents(int row, int col);
  void postSegment(obsdet_msgs::CloudClusterArray &in_out_cluster_array);
  void clusterIndices_Trans();
  void calculate_index2rc(int index, int &r, int &c);

  void eval_OSE();
  void eval_USE();

  void eval_running_time(int running_time);

  void MainLoop(const sensor_msgs::PointCloud2::ConstPtr& lidar_points);
  void publishSegmentedCloudsColor(const std_msgs::Header& header);

 public:
  FCC();
  ~FCC() {};

  void allocateMemory(){
    laserCloudIn.reset(new pcl::PointCloud<PointType>());
    segment_visul.reset(new pcl::PointCloud<pcl::PointXYZI>());
    fullCloud.reset(new pcl::PointCloud<PointType>());
    windowCloud.reset(new pcl::PointCloud<PointType>());

    // use_vector.clear();
    // runtime_vector.clear();

    fullCloud->points.resize(VERT_SCAN * HORZ_SCAN);
    windowCloud->points.resize(WINDOW_H * WINDOW_W);

    index_v.resize(VERT_SCAN * HORZ_SCAN);
  }

  void resetParameters(){
    laserCloudIn->clear();
            
    segment_visul->clear();
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(windowCloud->points.begin(), windowCloud->points.end(), nanPoint);

    labelMat = cv::Mat(VERT_SCAN, HORZ_SCAN, CV_32S, cv::Scalar::all(-1));
    rangeMat = cv::Mat(VERT_SCAN, HORZ_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    clusterIndices.clear();
    clusterIndices_ri.clear();
    gt_clusterIndices.clear();
    index_v.clear();
  }
};

// Dynamic parameter server callback function
void dynamicParamCallback(fcc::fcc_Config& config, uint32_t level)
{
  // Pointcloud Filtering Parameters
  th_H=config.segmentTh_H;
  th_V=config.segmentTh_V;
}

FCC::FCC():private_nh("~")
{
  std::string cloud_ground_topic;
  std::string jsk_bboxes_topic;

  private_nh.param<std::string>("lidar_points_topic", lidar_points_topic_, "/semi_kitti/non_ground_pc");
  private_nh.param<std::string>("output_frame", output_frame_, "map");
  private_nh.param<int>("window_h", WINDOW_H, 7);
  private_nh.param<int>("window_w", WINDOW_W, 7);

  allocateMemory();

  float resolution = (float)(MAX_VERT_ANGLE - MIN_VERT_ANGLE) / (float)(VERT_SCAN - 1);
  for (int i = 0; i < VERT_SCAN; i++)
    vert_angles_.push_back(MIN_VERT_ANGLE + i * resolution);

  sub_lidar_points = nh.subscribe(lidar_points_topic_, 1, &FCC::MainLoop, this);
  pub_segment_visul_ = nh.advertise<sensor_msgs::PointCloud2> ("/clustering/colored_cluster", 1);

  pub_test_visul_ = nh.advertise<sensor_msgs::PointCloud2>("/clustering/test_cluster", 1);

  pub_obsdet_clusters_ = nh.advertise<obsdet_msgs::CloudClusterArray>("/clustering/cluster_array", 1);
  pub_jskrviz_time_ = nh.advertise<std_msgs::Float32>("/total_time", 1);

  // Dynamic Parameter Server & Function
  f = boost::bind(&dynamicParamCallback, _1, _2);
  server.setCallback(f);

  // Create point processor
  nanPoint.x = std::numeric_limits<float>::quiet_NaN();
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();
  nanPoint.intensity = -1;
  nanPoint.id = 0;
  nanPoint.label = 0;
  nanPoint.cid = 9999;

  resetParameters();
}

int FCC::getRowIdx(PointType pt)
{
  float angle = atan2(pt.z, sqrt(pt.x * pt.x + pt.y * pt.y)) * 180 / M_PI;

  auto iter_geq = std::lower_bound(vert_angles_.begin(), vert_angles_.end(), angle);
  int row_idx;

  if (iter_geq == vert_angles_.begin())
  {
    row_idx = 0;
  }
  else
  {
    float a = *(iter_geq - 1);
    float b = *(iter_geq);
    if (fabs(angle - a) < fabs(angle - b))
    {
      row_idx = iter_geq - vert_angles_.begin() - 1;
    }
    else
    {
      row_idx = iter_geq - vert_angles_.begin();
    }
  }
  return row_idx;
}

int FCC::getColIdx(PointType pt)
{
  float horizonAngle = atan2(pt.x, pt.y) * 180 / M_PI;
  static float ang_res_x = 360.0 / float(HORZ_SCAN);
  int col_idx = -round((horizonAngle - 90.0) / ang_res_x) + HORZ_SCAN / 2;
  if (col_idx >= HORZ_SCAN)
    col_idx -= HORZ_SCAN;
  return col_idx;
}

void FCC::sphericalProjection(const sensor_msgs::PointCloud2::ConstPtr& laserRosCloudMsg) {
  float range;
  size_t rowIdn, columnIdn, index, cloudSize; 
  PointType curpt;
  pcl::PointXYZI point_ring;

  pcl::PointCloud<PointXYZILID>::Ptr raw_pcl(new pcl::PointCloud<PointXYZILID>);

  cloudHeader = laserRosCloudMsg->header;
  cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
  pcl::fromROSMsg(*laserRosCloudMsg, *raw_pcl);

  cloudSize = raw_pcl->points.size();

  int scan_num=0;
  int oindex = 0;
  laserCloudIn->clear();

  for (size_t i = 0; i < cloudSize; ++i)
  {
    curpt.x = raw_pcl->points[i].x;
    curpt.y = raw_pcl->points[i].y;
    curpt.z = raw_pcl->points[i].z;
    curpt.intensity = raw_pcl->points[i].intensity;
    curpt.ring = raw_pcl->points[i].ring;
    curpt.id = raw_pcl->points[i].id;
    curpt.label = raw_pcl->points[i].label;
    curpt.cid = 9999;

    bool is_nan = std::isnan(curpt.x) || std::isnan(curpt.y) || std::isnan(curpt.z);
    if (is_nan)
      continue;

    //find the row and column index in the iamge for this point
    rowIdn = getRowIdx(curpt);

    if (rowIdn < 0 || rowIdn >= VERT_SCAN)
      continue;

    columnIdn = getColIdx(curpt);

    if (columnIdn < 0 || columnIdn >= HORZ_SCAN)
      continue;

    range = sqrt(curpt.x * curpt.x + curpt.y * curpt.y + curpt.z * curpt.z);

    // if (range > 10)
    //   continue;

    labelMat.at<int>(rowIdn, columnIdn) = 0;
    rangeMat.at<float>(rowIdn, columnIdn) = range;

    index = columnIdn  + rowIdn * HORZ_SCAN;

    fullCloud->points[index] = curpt;

    index_v[index] = oindex;

    laserCloudIn->points.push_back(curpt);
    oindex++;
  }
}

void FCC::postSegment(obsdet_msgs::CloudClusterArray &in_out_cluster_array)
{
  unsigned int intensity_mark = 1;
  
  pcl::PointXYZI cluster_color;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_pcl(new pcl::PointCloud<pcl::PointXYZI>);

  for (auto& getIndices : clusterIndices)
  {
    cluster_pcl->clear();
    // laserCloudIn  fullCloud
    for (auto& index : getIndices){
      cluster_color.x = laserCloudIn->points[index].x;
      cluster_color.y = laserCloudIn->points[index].y;
      cluster_color.z = laserCloudIn->points[index].z;
      cluster_color.intensity = intensity_mark;

      cluster_pcl->push_back(cluster_color);
      segment_visul->push_back(cluster_color);
    }

    intensity_mark++;
    cloud_msg.header = _velodyne_header;
    pcl::toROSMsg(*cluster_pcl, cloud_msg);

    obsdet_msgs::CloudCluster cluster_;
    cluster_.header = _velodyne_header;
    cluster_.cloud = cloud_msg;
    in_out_cluster_array.clusters.push_back(cluster_);
  }

  sensor_msgs::PointCloud2 segment_visul_ros;
  pcl::toROSMsg(*segment_visul, segment_visul_ros);
  segment_visul_ros.header = _velodyne_header;
  pub_segment_visul_.publish(segment_visul_ros);

  in_out_cluster_array.header = _velodyne_header;
  pub_obsdet_clusters_.publish(in_out_cluster_array);
}

void FCC::clusterIndices_Trans()
{
  int row, col;
  std::vector<int> clusterIndice;

  for (auto &getIndices : clusterIndices_ri)
  {
    clusterIndice.clear();
    for (auto &index : getIndices)
    {
      calculate_index2rc(index, row, col);
      clusterIndice.push_back(index_v[col + row * HORZ_SCAN]);
    }
    clusterIndices.push_back(clusterIndice);
  }
}

void FCC::calculate_index2rc(int index, int &r, int &c){
  int j;
  for (int i = 0; i < VERT_SCAN; ++i)
  {
    j = index - i * HORZ_SCAN;
    if (j <= HORZ_SCAN){
      r=i;
      c=j;
      break;
    }
  }
}

void FCC::eval_OSE()
{
  double ose_i = 0.0;

  int cluster_id = 0;
  for (auto &getIndices : clusterIndices)
  {
    for (auto &index : getIndices)
    {
      laserCloudIn->points[index].cid = cluster_id;
    }
    cluster_id++;
  }

  for (auto &getIndices : gt_clusterIndices)
  {
    int object_cluster[clusterIndices.size()] = {0};
    int N = getIndices.size();

    for (auto &index : getIndices)
    {
      if (laserCloudIn->points[index].cid != 9999)
        object_cluster[laserCloudIn->points[index].cid]++;
    }

    for (size_t i = 0; i < clusterIndices.size(); i++)
    {
      if (object_cluster[i] == 0)
        continue;

      double ose_ii = -(object_cluster[i] / (1.0*N)) * log(object_cluster[i] / (1.0*N));

      ose_i += ose_ii;
    }
  }

  ose_vector.push_back(ose_i);

  std::cout << "ose_vector.size() is = " << ose_vector.size() << std::endl;

  double ose_total_v = 0.0;
  double ose_sqr_sum = 0.0;
  double ose_mean;
  double ose_std;

  for (size_t i = 0; i < ose_vector.size(); i++)
  {
    ose_total_v += ose_vector[i];
  }
  ose_mean = ose_total_v / ose_vector.size();

  for (size_t i = 0; i < ose_vector.size(); i++)
  {
    ose_sqr_sum += (ose_vector[i] - ose_mean) * (ose_vector[i] - ose_mean);
  }

  ose_std = sqrt(ose_sqr_sum / ose_vector.size());

  std::cout << "current ose_i is = " << ose_i << std::endl;
  std::cout << "\033[1;34mose_mean is = " << ose_mean << "\033[0m" << std::endl;
  std::cout << "ose_std is = " << ose_std << std::endl;
}

void FCC::eval_USE()
{
  std::vector<int> cluater_label;
  std::vector<std::vector<int> > cluaters_label;

  int label[34] = {0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51,
                   52, 60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259};

  double use_i_sum = 0;

  for (auto& getIndices : clusterIndices)
  {
    int cluster_label[34] = {0};
    for (auto& index : getIndices)
    {
      for (size_t i = 0; i < 34; i++)
      {
        if (laserCloudIn->points[index].label == label[i])
          cluster_label[i]++;
      }
    }

    int M = getIndices.size();

    for (size_t i = 0; i < 34; i++)
    {
      if (cluster_label[i] == 0)
        continue;

      double use_i = -(cluster_label[i] / (M * 1.0)) * log(cluster_label[i] / (M * 1.0));
      use_i_sum += use_i;
    }
  }

  use_vector.push_back(use_i_sum);  

  double use_total_v = 0.0;
  double use_sqr_sum = 0.0;
  double use_mean;
  double use_std;

  ROS_INFO("use_vector.size()=%d",use_vector.size());

  for (size_t i = 0; i < use_vector.size(); i++)
  {
    use_total_v += use_vector[i];
  }
  use_mean = use_total_v / use_vector.size();

  for (size_t i = 0; i < use_vector.size(); i++)
  {
    use_sqr_sum += (use_vector[i] - use_mean) * (use_vector[i] - use_mean);
  }

  use_std = sqrt(use_sqr_sum / use_vector.size());

  std::cout << "current use_i is = " << use_i_sum << std::endl;
  std::cout << "\033[1;32muse_mean is = " << use_mean << "\033[0m" << std::endl;
  std::cout << "use_std is = " << use_std << std::endl;
}

void FCC::eval_running_time(int running_time)
{
  double runtime_std;
  double runtime_aver;
  double runtime_total_v = 0.0;
  double runtime_sqr_sum = 0.0;

  runtime_vector.push_back(running_time);

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_total_v += runtime_vector[i];
  }

  runtime_aver = runtime_total_v / runtime_vector.size();

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_sqr_sum += (runtime_vector[i] - runtime_aver) * (runtime_vector[i] - runtime_aver);
  }

  runtime_std = sqrt(runtime_sqr_sum / runtime_vector.size());

  std::cout << "runtime_vector.size() is = " << runtime_vector.size() << std::endl;
  std::cout << "current running_time is = " << running_time << "ms" << std::endl;
  std::cout << "\033[1;36mruntime_aver is = " << runtime_aver << "ms"
            << "\033[0m" << std::endl;
  std::cout << "runtime_std is = " << runtime_std << "ms" << std::endl;
}


void FCC::MainLoop(const sensor_msgs::PointCloud2::ConstPtr& lidar_points)
{
  total_start_time_ = std::chrono::system_clock::now();

  _velodyne_header = lidar_points->header;

  obsdet_msgs::CloudClusterArray cluster_array;
  gt_verify.reset(new groundtruth::DepthCluster<PointType>());

  sphericalProjection(lidar_points);

  // segmentation process
  const auto start_time = std::chrono::steady_clock::now();

  // double th_H = 0.35;
  // double th_V = 1.0;
  int maxlabel_lines[64] = {0};
  bool valid_ring[64] = {false};

  for (size_t i = 0; i < 64; i++)
  {
    maxlabel_lines[i] = 1;
  }

  PointType point_current;
  PointType point_first;
  PointType point_neighbor;
  PointType point_last;

  /* ----------------------- first run label merging ----------------------- */
  for (size_t i = 0; i < VERT_SCAN; ++i)
  {
    bool valid_first = false;

    // maxlabel = 1
    for (size_t j = 0; j < HORZ_SCAN; ++j)
    {
      if (labelMat.at<int>(i, j) == -1)
        continue;

      // get the current point
      point_current = fullCloud->points[(j + i * HORZ_SCAN)];

      if (!valid_first)
      {
        point_first = point_current;
        point_neighbor = point_current;
        valid_first = true;
      }

      // group the line wise point
      // if valid_neighbor:
      double d_cn;
      d_cn = (point_current.x - point_neighbor.x) * (point_current.x - point_neighbor.x) + (point_current.y - point_neighbor.y) * (point_current.y - point_neighbor.y) +
             (point_current.z - point_neighbor.z) * (point_current.z - point_neighbor.z);
      if (sqrt(d_cn) <= th_H)
        labelMat.at<int>(i, j) = maxlabel_lines[i];
      else
      {
        maxlabel_lines[i] = maxlabel_lines[i] + 1;
        labelMat.at<int>(i, j) = maxlabel_lines[i];
      }

      // update the point neighbor and point last
      point_neighbor = point_current;
      point_last = point_current;
    }

    // note the maxlabel for each line and ring connection
    double d_f2l;
    d_f2l = (point_first.x - point_last.x) * (point_first.x - point_last.x) + (point_first.y - point_last.y) * (point_first.y - point_last.y) +
            (point_first.z - point_last.z) * (point_first.z - point_last.z);
    if (sqrt(d_f2l) <= th_H)
      valid_ring[i] = true;
    else
      valid_ring[i] = false;
  }

  // Accumulate the labels
  int start_label = 0;
  int max_label = -9999;

  for (size_t i = 0; i < VERT_SCAN; ++i)
  {
    if (valid_ring[i])
      maxlabel_lines[i] = maxlabel_lines[i] - 1;

    for (size_t j = 0; j < HORZ_SCAN; ++j)
    {
      if (labelMat.at<int>(i, j) == -1)
        continue;

      if (valid_ring[i])
      {
        if (labelMat.at<int>(i, j) > maxlabel_lines[i])
          labelMat.at<int>(i, j) = 1;
      }

      labelMat.at<int>(i, j) = labelMat.at<int>(i, j) + start_label;

      if (labelMat.at<int>(i, j) > max_label)
        max_label = labelMat.at<int>(i, j);
    }
    start_label = start_label + maxlabel_lines[i];
  }

  /* ----------------------- second run label merging ----------------------- */
  int window_size[2] = {WINDOW_H, WINDOW_W};

  int WINDOW_H_ = (WINDOW_H - 1) / 2;
  int WINDOW_W_ = (WINDOW_W - 1) / 2;

  // // init the merge variables
  int label_current = 1;
  std::vector<int> labelsToMerge;

  // mergeTable = np.arange(0, maxlabel + 1)
  int mergeTable[max_label + 2] = {0};
  for (size_t i = 0; i < max_label + 2; i++)
  {
    mergeTable[i] = i;
  }

  for (size_t h = 0; h < VERT_SCAN; ++h)
  {
    for (size_t w = 0; w < HORZ_SCAN; ++w)
    {
      if (labelMat.at<int>(h, w) == -1 && w != HORZ_SCAN - 1)
        continue;

      // update the window
      for (int i = 0; i < WINDOW_H; ++i)
      {
        for (int j = 0; j < WINDOW_W; ++j)
        {
          int offsetY = h + i - WINDOW_H_;
          int offsetX = w + j - WINDOW_W_;
          int offsetIdx = offsetX + offsetY * HORZ_SCAN;

          if (offsetY >= 0 && offsetY < VERT_SCAN && offsetX >= 0 && offsetX < HORZ_SCAN) {
            windowCloud->points[j + i * WINDOW_H] = fullCloud->points[offsetIdx];
          }
          else {
            windowCloud->points[j + i * WINDOW_H] = nanPoint;     
          }
        }
      }

      point_current = fullCloud->points[(w + h * HORZ_SCAN)];

      // update the labels if current label change or end of frame
      // # if (label_current != point_current[4] or (h == height-1 and w == width-1)):
      if (label_current != labelMat.at<int>(h, w) || w == HORZ_SCAN-1) {
        if (labelsToMerge.size() != 0) {
          //  Sort and unique
          sort(labelsToMerge.begin(), labelsToMerge.end());
          std::vector<int>::iterator pos = unique(labelsToMerge.begin(), labelsToMerge.end());
          labelsToMerge.erase(pos, labelsToMerge.end());

          // update the merge Table
          for (size_t n = 0; n < labelsToMerge.size(); n++)
          {
            mergeTable[labelsToMerge[n]] = mergeTable[labelsToMerge[0]];
          }
        }

        //  update the label_current
        //  print(label_current, point_current[4])
        label_current = labelMat.at<int>(h, w) ;
        labelsToMerge.clear();
      }

      for (size_t i = 0; i < window_size[0]; i++)
      {
        for (size_t j = 0; j < window_size[1]; j++)
        {
          point_neighbor = windowCloud->points[j + i * WINDOW_H];

          bool is_nan = std::isnan(point_neighbor.x) || std::isnan(point_neighbor.y) || std::isnan(point_neighbor.z);
          if (is_nan){continue;}
          double d_c2n;
          d_c2n = (point_current.x - point_neighbor.x) * (point_current.x - point_neighbor.x) + (point_current.y - point_neighbor.y) * (point_current.y - point_neighbor.y) +
                  (point_current.z - point_neighbor.z) * (point_current.z - point_neighbor.z);
          if (sqrt(d_c2n) < th_V)
          {
            std::pair<int /*point_current*/, int /*point_neighbor*/> label_pair;
            label_pair = std::make_pair(mergeTable[labelMat.at<int>(h, w)], mergeTable[labelMat.at<int>(h + i - WINDOW_H_, w + j - WINDOW_W_)]);

            sort(labelsToMerge.begin(), labelsToMerge.end());
            if (!std::binary_search(labelsToMerge.begin(), labelsToMerge.end(), label_pair.first))
              labelsToMerge.push_back(label_pair.first);
            sort(labelsToMerge.begin(), labelsToMerge.end());
            if (!std::binary_search(labelsToMerge.begin(), labelsToMerge.end(), label_pair.second))
              labelsToMerge.push_back(label_pair.second);
          }
          else
            {continue;}
        }
      }
    }
  }

  std::vector<std::vector<int> > cluster_idx;
  cluster_idx.resize(max_label + 2);

  for (size_t i = 0; i < VERT_SCAN; ++i)
  {
    for (size_t j = 0; j < HORZ_SCAN; ++j)
    {
      if (labelMat.at<int>(i, j) == -1)
        continue;

      int outLabel = mergeTable[labelMat.at<int>(i, j)];
      outLabel = mergeTable[outLabel];
      cluster_idx[outLabel].push_back(j + i * HORZ_SCAN);
    }
  }

  for (size_t i = 0; i < cluster_idx.size(); i++)
  {
    if (cluster_idx[i].size()>10)
    {
      std::vector<int> c_idx;
      for (size_t j = 0; j < cluster_idx[i].size(); j++)
      {
        c_idx.push_back(cluster_idx[i][j]);
      }
      clusterIndices_ri.push_back(c_idx);
    }
  }

  clusterIndices_Trans();

  // Time the whole process
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  eval_running_time(elapsed_time.count());

  gt_verify->GTV(laserCloudIn, gt_clusterIndices);

  eval_OSE();
  eval_USE();

  //visualization, use indensity to show different color for each cluster.
  postSegment(cluster_array);

  resetParameters();

  // total_end_time_ = std::chrono::system_clock::now();
  // total_exe_time_ = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time_ - total_start_time_).count() / 1000.0;
  time_spent.data = elapsed_time.count();
  pub_jskrviz_time_.publish(time_spent);

  std::cout << "---------------------------------" << std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "fcc_node");

  tf::StampedTransform transform;
  tf::TransformListener listener;

  _transform = &transform;
  _transform_listener = &listener;

  FCC FCC_node;

  ros::spin();

  return 0;
}