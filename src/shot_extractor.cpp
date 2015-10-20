#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <iostream>

#include "load_obj.hpp"
#include "obj_to_pc.hpp"

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

// dimension of the different descriptors
#define DIM_SHOT 352
#define DIM_REF 9

std::string model_filename_;
std::string output_filename_;

// Algorithm static params
bool use_cloud_resolution_ (true);

// values for resolution == 1.0f
float model_ss_ (2.5f);
float descr_rad_ (100.0f);
int normals_nn_ (100);

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h")) {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> in_filenames;
  std::vector<int> out_filenames;
  in_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".obj");
  out_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ftr");

  if (in_filenames.size () != 1) {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }
  model_filename_ = argv[in_filenames[0]];

  if (out_filenames.size () == 1) {
    output_filename_ = argv[out_filenames[0]];
  }

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-r")) {
    use_cloud_resolution_ = true;
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  int nn = 20;
  std::vector<int> indices (nn);
  std::vector<float> sqr_distances (nn);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i) {
    if (! pcl_isfinite ((*cloud)[i].x)) {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, nn, indices, sqr_distances);
    if (nres == nn) {
      int j = 0;
      while (j < sqr_distances.size() && sqr_distances[j] <= 0) {
        j++;
      }
      if (j < sqr_distances.size() && !isnan(sqr_distances[j])) {
        res += sqrt (sqr_distances[j]);
        ++n_points;
      }
    }
  }
  if (n_points != 0) {
    res /= n_points;
  }
  return res;
}

int
main (int argc, char *argv[])
{
  parseCommandLine (argc, argv);

  // init pointclouds
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  
  // load all points and triangles from the obj file
  std::vector< std::vector<float> >  obj_pts;
  std::vector< std::vector<float> >   obj_tris;

  // load obj and convert to pointcloud
  LoadOBJFile(model_filename_.c_str(), obj_pts, obj_tris);
  for (int ii = 0; ii < obj_pts.size (); ii++) {
    pcl::PointXYZ p(obj_pts[ii][0],obj_pts[ii][1],obj_pts[ii][2]); 
    (*model).push_back(p);
  }

  // use the cloud resolution to generate points
  if (use_cloud_resolution_) {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f) {
      model_ss_   *= resolution;
      descr_rad_  *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
  }

  // estimate normals
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch(normals_nn_);
  norm_est.setInputCloud(model);
  norm_est.compute(*model_normals);

  // subsample points uniformly on the surface
  pcl::PointCloud<int> sampled_indices;
  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  // get SHOT descriptors
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);
  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  // pcl::KdTreeFLANN<DescriptorType> match_search;
  // match_search.setInputCloud (model_descriptors);
  // std::cout << "Dims " << match_search.getPointRepresentation()->getNumberOfDimensions() << std::endl;
  // std::cout << "Is Trivial? " << match_search.getPointRepresentation()->isTrivial() << std::endl;

  // float p[352];
  // match_search.getPointRepresentation()->copyToFloatArray(model_descriptors->at(0), p);
  // for (int i = 0; i < 352; i++) {
  //   float diff = p[i] - model_descriptors->points[0].descriptor[i];
  //   if (diff > 0)
  //     std::cout << "Weird " << i << " has diff " << diff << std::endl;
  // }

  // open output file
  std::ofstream outf(output_filename_.c_str());
  if (!outf.is_open()) {
    std::cerr << output_filename_ << " could not be opened for writing" << std::endl;
    exit(1);
  }
  
  // write header
  outf << model_keypoints->points.size () << std::endl;
  outf << DIM_SHOT << std::endl;
  outf << DIM_REF << std::endl;

  // for each point write
  for (int ii = 0; ii < model_keypoints->points.size (); ii++) {
    // reference frame
    for (int jj = 0; jj < DIM_REF; jj++) {
      outf << (model_descriptors->points[ii]).rf[jj] << " " ;
    }
    outf << "\t";

    // shot descriptor
    for (int jj = 0; jj < DIM_SHOT; jj++) {
      outf << (model_descriptors->points[ii]).descriptor[jj] << " " ;
    }
    outf << "\t";

    // point
    outf << model_keypoints->points[ii].x << " " << model_keypoints->points[ii].y << " " << model_keypoints->points[ii].z << " \t";

    // normal
    for (int jj = 0; jj < model_normals->points.size (); jj++) {
      if ((model->points[jj].x == model_keypoints->points[ii].x) && (model->points[jj].y == model_keypoints->points[ii].y)) {
        outf << model_normals->points[jj].normal_x << " " << model_normals->points[jj].normal_y << " " << model_normals->points[jj].normal_z  << std::endl;
        break;
      }
    }
  }

  outf.close();
  return (0);
}


