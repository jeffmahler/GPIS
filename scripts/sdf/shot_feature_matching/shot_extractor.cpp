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
#define DIM_SHOT 352
#define DIM_REF 9

std::string model_filename_;
std::string output_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

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
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> in_filenames;
  std::vector<int> out_filenames;
  in_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".obj");
  out_filenames = pcl::console::parse_file_extension_argument (argc, argv, ".txt");
  if (in_filenames.size () != 1)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }
  model_filename_ = argv[in_filenames[0]];

  if (out_filenames.size () == 1)
  {
    output_filename_ = argv[out_filenames[0]];
  }

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

int
main (int argc, char *argv[])
{
  parseCommandLine (argc, argv);

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  
  //load all points and triangles from the obj file
  std::vector< std::vector<float> >  obj_pts;
  std::vector< std::vector<float> >   obj_tris;
 // OBJtoPC(model_filename_.c_str(), *model);

  std::cout << model_filename_.c_str() << std::endl;
  LoadOBJFile(model_filename_.c_str(), obj_pts, obj_tris);
  for (int ii = 0; ii < obj_pts.size (); ii++)
    {
        pcl::PointXYZ p(obj_pts[ii][0],obj_pts[ii][1],obj_pts[ii][2]); 
        (*model).push_back(p);
    }


  if (use_cloud_resolution_)
  {
    float resolution = static_cast<float> (computeCloudResolution (model));
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  }

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  pcl::PointCloud<int> sampled_indices;

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;

  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  //std::cout << model_descriptors->points.size () << std::endl;
  //std::cout << model_normals->points.size () << std::endl;
  //std::cout << model_keypoints->points.size () << std::endl;
  //std::cout << model->points[2031-9] << std::endl;
  //std::cout << model_keypoints->points[0] << std::endl;
  //for (int kk = 0; kk < 352; kk++)
  //  std::cout << ((model_descriptors->points[1]).descriptor).size () << " ";

  std::ofstream outf;
  outf.open (output_filename_.c_str(), std::ios::in | std::ios::out);

  if (!outf)
  {
    std::cerr << output_filename_ << " could not be opened for writing" << std::endl;
    exit(1);
  }
  
  outf << model_keypoints->points.size () << std::endl;
  outf << DIM_SHOT << std::endl;
  outf << DIM_REF << std::endl;

    
  for (int ii = 0; ii < model_keypoints->points.size (); ii++)
  {
    //outf << model_descriptors->points[ii] << " \t" ;
    for (int jj = 0; jj < DIM_REF; jj++)
    {
      outf << (model_descriptors->points[ii]).rf[jj] << " " ;
    }
    outf << "\t";
    for (int jj = 0; jj < DIM_SHOT; jj++)
    {
      outf << (model_descriptors->points[ii]).descriptor[jj] << " " ;
    }
    outf << "\t";
    outf << model_keypoints->points[ii].x << " " << model_keypoints->points[ii].y << " " << model_keypoints->points[ii].z << " \t";
    for (int jj = 0; jj < model_normals->points.size (); jj++)
    {
      if ((model->points[jj].x == model_keypoints->points[ii].x) && (model->points[jj].y == model_keypoints->points[ii].y)
      {
        outf << model_normals->points[jj].normal_x << " " << model_normals->points[jj].normal_y << " " << model_normals->points[jj].normal_z  << std::endl;
        break;
      }
    }
  }

  outf.close();

  return (0);
}


