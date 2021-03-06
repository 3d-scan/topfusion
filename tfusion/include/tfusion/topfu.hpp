#pragma once

#include <tfusion/types.hpp>
// #include <tfusion/cuda/tsdf_volume.hpp>
#include <tfusion/cuda/projective_icp.hpp>
#include <vector>
#include <string>
// #include <tfusion/cuda/reconstruction.hpp>
#include <tfusion/Defines.hpp>
#include <tfusion/cuda/SceneReconstructionEngine_host.hpp>
#include <tfusion/cuda/VisualisationEngine_CUDA.hpp>
#include <tfusion/scene.hpp>
#include <tfusion/RenderState.hpp>
#include <tfusion/RenderState_VH.hpp>

namespace tfusion
{
    namespace cuda
    {
         int getCudaEnabledDeviceCount();
         void setDevice(int device);
         std::string getDeviceName(int device);
         bool checkIfPreFermiGPU(int device);
         void printCudaDeviceInfo(int device);
         void printShortCudaDeviceInfo(int device);
    }

    struct KF_EXPORTS TopFuParams
    {
        static TopFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

        SceneParams *sceneParams;
    };

    class KF_EXPORTS TopFu
    {
    public:        
        typedef cv::Ptr<TopFu> Ptr;

        TopFu(const TopFuParams& params);

        const TopFuParams& params() const;
        TopFuParams& params();

        // const cuda::TsdfVolume& tsdf() const;
        // cuda::TsdfVolume& tsdf();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        void reset();

        bool operator()(const cuda::Depth& dpeth, const cuda::Image& image = cuda::Image());

        // void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::image4u& image);//, const Affine3f& pose, int flags = 0);

        Affine3f getCameraPose (int time = -1) const;
    private:
        void allocate_buffers();

        int frame_counter_;
        TopFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        // cv::Ptr<cuda::TsdfVolume> volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;

        // View *view;
        // ViewBuilder *viewBuilder;
        Scene<Voxel_s,VoxelBlockHash> *scene;
        SceneReconstructionEngine_CUDA<Voxel_s,VoxelBlockHash> *sceneEngine;
        RenderState *renderState;
        VisualisationEngine_CUDA<Voxel_s,VoxelBlockHash> *visualisationEngine;
    };
}