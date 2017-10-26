#include "precomp.hpp"
#include "internal.hpp"
#include "tfusion/topfu.hpp"
#include "tfusion/cuda/SceneReconstructionEngine_host.hpp"

using namespace std;
using namespace tfusion;
using namespace tfusion::cuda;

static inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

tfusion::TopFuParams tfusion::TopFuParams::default_params()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    TopFuParams p;

    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.intr = Intr(525.f, 525.f, p.cols/2 - 0.5f, p.rows/2 - 0.5f);
	//p.intr = Intr(580.f,580.f,320.f,240.f);
	//p.intr = Intr(573.71,574.394,346.471,249.031);
	p.intr = Intr(504.261,503.905,352.457,272.202);

    p.volume_dims = Vec3i::all(512);  //number of voxels
    p.volume_size = Vec3f::all(3.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    //p.icp_truncate_depth_dist = 0.f;        //meters, disabled
	p.icp_truncate_depth_dist = 2.0f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    p.sceneParams = new tfusion::SceneParams(0.02f, 100, 0.005f, 0.2f, 3.0f, false);

    return p;
}

tfusion::TopFu::TopFu(const TopFuParams& params) : frame_counter_(0), params_(params)
{
    // CV_Assert(params.volume_dims[0] % 32 == 0);

    // volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));

    // volume_->setTruncDist(params_.tsdf_trunc_dist);
    // volume_->setMaxWeight(params_.tsdf_max_weight);
    // volume_->setSize(params_.volume_size);
    // volume_->setPose(params_.volume_pose);
    // volume_->setRaycastStepFactor(params_.raycast_step_factor);
    // volume_->setGradientDeltaFactor(params_.gradient_delta_factor);
    scene = new Scene<Voxel_s,VoxelBlockHash>(params_.sceneParams,false);
    // sceneEngine = dynamic_cast<tfusion::SceneReconstructionEngine_CUDA<Voxel_s,VoxelBlockHash>*>(new SceneReconstructionEngine_CUDA<Voxel_s,VoxelBlockHash>);
    sceneEngine = new SceneReconstructionEngine_CUDA<Voxel_s,VoxelBlockHash>;
    tfusion::RenderState::Vector2i_host imgSize(params_.cols,params_.rows);
    renderState = new RenderState_VH(VoxelBlockHash::noTotalEntries, imgSize,params_.sceneParams->viewFrustum_min,params_.sceneParams->viewFrustum_max);
    visualisationEngine = new VisualisationEngine_CUDA<Voxel_s,VoxelBlockHash>;
// hello();
	//cout<<"renderstate cols :"<<renderState->renderingRangeImage.cols()<<",rows:"<<renderState->renderingRangeImage.rows()<<endl;
    sceneEngine->ResetScene(scene);

    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    allocate_buffers();
    reset();
}

const tfusion::TopFuParams& tfusion::TopFu::params() const
{ return params_; }

tfusion::TopFuParams& tfusion::TopFu::params()
{ return params_; }

// const tfusion::cuda::TsdfVolume& tfusion::TopFu::tsdf() const
// { return *volume_; }

// tfusion::cuda::TsdfVolume& tfusion::TopFu::tsdf()
// { return *volume_; }

const tfusion::cuda::ProjectiveICP& tfusion::TopFu::icp() const
{ return *icp_; }

tfusion::cuda::ProjectiveICP& tfusion::TopFu::icp()
{ return *icp_; }

void tfusion::TopFu::allocate_buffers()
{
    const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void tfusion::TopFu::reset()
{
    if (frame_counter_)
        cout << "Reset" << endl;

    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(30000);
    poses_.push_back(Affine3f::Identity());
    // volume_->clear();
	sceneEngine->ResetScene(scene);
}

tfusion::Affine3f tfusion::TopFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}

bool tfusion::TopFu::operator()(const tfusion::cuda::Depth& depth,const tfusion::cuda::Image&)
{
    const TopFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();

    cuda::computeDists(depth,dists_,p.intr);
    cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);
	//depth.copyTo(curr_.depth_pyr[0]);
	//cuda::computeDists(curr_.depth_pyr[0],dists_,p.intr);
	/*cv::Mat depth_raw(480,640,CV_16U);
	cv::Mat depth_filter(480,640,CV_16U);

	depth.download(depth_raw.data,depth_raw.step);
	curr_.depth_pyr[0].download(depth_filter.data,depth_filter.step);*/
	
	//cv::Mat dist_display(dists_.cols(),dists_.rows(),CV_32F);
	//cv::Mat dist_display(dists_.rows(),dists_.cols(),CV_32F);
	/*cv::Mat dist_display(p.rows,p.cols,CV_32F);
	cout<<"step:"<<dist_display.step<<endl;
	dists_.download(dist_display.data,dist_display.step);*/
	
	/*cv::Mat dist_char(p.rows,p.cols,CV_8UC1);
	int temp;
	for(int i=0;i<480;i++)
		for(int j=0;j<640;j++)
		{
			temp = (int)(dist_display.data[j + i * 640]);
			dist_char.data[j + i * 640] = (uchar)(temp) * 10;
		}*/
    if(p.icp_truncate_depth_dist > 0)
        tfusion::cuda::depthTruncation(curr_.depth_pyr[0],p.icp_truncate_depth_dist);

    for(int i = 1;i < LEVELS;i++)
        cuda::depthBuildPyramid(curr_.depth_pyr[i-1],curr_.depth_pyr[i],p.bilateral_sigma_depth);

    for(int i=0;i < LEVELS;i++)
        cuda::computePointNormals(p.intr(i),curr_.depth_pyr[i],curr_.points_pyr[i],curr_.normals_pyr[i]);
    cuda::waitAllDefaultStream();

    if(frame_counter_ == 0)
    {
		sceneEngine->AllocateSceneFromDepth(scene,p.intr,poses_.back(),dists_,renderState);
        sceneEngine->IntegrateIntoScene(scene,p.intr,poses_.back(),dists_,renderState);
        // curr_.depth_pyr.swap(prev_.depth_pyr);
        curr_.points_pyr.swap(prev_.points_pyr);

        curr_.normals_pyr.swap(prev_.normals_pyr);
        return ++frame_counter_,true;
    }

    Affine3f affine;
	cv::Mat points_mat(480,640,CV_32FC4);
	curr_.points_pyr[0].download(points_mat.data,points_mat.step);

	cv::Mat points_mat1(480,640,CV_32FC4);
	prev_.points_pyr[0].download(points_mat1.data,points_mat1.step);

	cv::Mat points_mat2(480,640,CV_32FC1);

	dists_.download(points_mat2.data,points_mat2.step);
	cv::Mat points_mat3(480,640,CV_16UC1);

	curr_.depth_pyr[0].download(points_mat3.data,points_mat3.step);

	//ifstream inFile;
	//inFile.open("D:\\a.txt",ios::in);
	////float a;
	//istream_iterator<float> begin(inFile);    //按 float 格式取文件数据流的起始指针  
 //   istream_iterator<float> end;          //取文件流的终止位置  
 //   vector<float> inData(begin,end);      //将文件数据保存至 std::vector 中  
	//for(int i=0;i<4;i++)
	//	for(int j=0;j<4;j++)
	//	{
	//		affine.matrix(i,j) = inData[(frame_counter_-1)*16 + i * 4 + j];
	//	}

	//cout<<affine.matrix(0,0)<<" "<<affine.matrix(0,1)<<" "<<affine.matrix(0,2)<<" "<<affine.matrix(0,3)<<endl
	//	<<affine.matrix(1,0)<<" "<<affine.matrix(1,1)<<" "<<affine.matrix(1,2)<<" "<<affine.matrix(1,3)<<endl
	//	<<affine.matrix(2,0)<<" "<<affine.matrix(2,1)<<" "<<affine.matrix(2,2)<<" "<<affine.matrix(2,3)<<endl
	//	<<affine.matrix(3,0)<<" "<<affine.matrix(3,1)<<" "<<affine.matrix(3,2)<<" "<<affine.matrix(3,3)<<endl;
    // bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
    bool ok = icp_->estimateTransform(affine,p.intr,curr_.points_pyr,curr_.normals_pyr,prev_.points_pyr,prev_.normals_pyr);
	poses_.push_back(poses_.back() * affine);
	//poses_.push_back(affine);
	Affine3f pose = poses_.back();
	Matrix4f M_d_print(pose.matrix(0,0),pose.matrix(1,0),pose.matrix(2,0),pose.matrix(3,0),
		pose.matrix(0,1),pose.matrix(1,1),pose.matrix(2,1),pose.matrix(3,1),
		pose.matrix(0,2),pose.matrix(1,2),pose.matrix(2,2),pose.matrix(3,2),
		pose.matrix(0,3),pose.matrix(1,3),pose.matrix(2,3),pose.matrix(3,3));


	cout<<"pose:"<<endl<<M_d_print<<endl;
	//cout<<"pose-before:"<<endl<<affine.matrix<<endl;
	/*Matrix4f a(1.f,0.f,0.f,0.f,
			   2.f,1.f,0.f,0.f,
			   0.f,0.f,1.f,0.f,
			   0.f,0.f,0.f,1.f);
	Matrix4f b(1.f,1.f,0.f,0.f,
			   0.f,1.f,0.f,0.f,
			   0.f,0.f,1.f,0.f,
			   0.f,0.f,0.f,1.f);
	cout<<"result:\n"<<a*b<<endl;*/
    if(!ok)
        return reset(),false;

    /*poses_.push_back(poses_.back() * affine);

    Affine3f pose = poses_.back();*/

    // Matrix4f M_d(pose.matrix(0,0),pose.matrix(0,1),pose.matrix(0,2),pose.matrix(0,3),
    //             pose.matrix(1,0),pose.matrix(1,1),pose.matrix(1,2),pose.matrix(1,3),
    //             pose.matrix(2,0),pose.matrix(2,1),pose.matrix(2,2),pose.matrix(2,3),
    //             pose.matrix(3,0),pose.matrix(3,1),pose.matrix(3,2),pose.matrix(3,3));
    //allocation and integration
	/*float rnorm = (float)cv::norm(affine.rvec());
    float tnorm = (float)cv::norm(affine.translation());
    bool integrate = (rnorm + tnorm)/2 >= p.tsdf_min_camera_movement;
	if(integrate)*/
    {
        // sceneEngine->AllocateSceneFromDepth(scene,view,pose_.back(),dists_);
        sceneEngine->AllocateSceneFromDepth(scene,p.intr,pose.inv(),dists_,renderState);
        sceneEngine->IntegrateIntoScene(scene,p.intr,pose.inv(),dists_,renderState);
    }
	cuda::image4u image;
	renderImage(image);
	cv::Mat points_mat6(480,640,CV_8UC4);

	image.download(points_mat6.data,points_mat6.step); 
    //prepare icp
    //visualisationEngine->CreateExpectedDepths(scene,pose,p.intr,renderState);

	Affine3f pose_inv = poses_.back();
	Matrix4f M_d(pose_inv.matrix(0,0),pose_inv.matrix(1,0),pose_inv.matrix(2,0),pose_inv.matrix(3,0),
		pose_inv.matrix(0,1),pose_inv.matrix(1,1),pose_inv.matrix(2,1),pose_inv.matrix(3,1),
		pose_inv.matrix(0,2),pose_inv.matrix(1,2),pose_inv.matrix(2,2),pose_inv.matrix(3,2),
		pose_inv.matrix(0,3),pose_inv.matrix(1,3),pose_inv.matrix(2,3),pose_inv.matrix(3,3));

	//Matrix4f M_d(pose.matrix(0,0),pose.matrix(1,0),pose.matrix(2,0),pose.matrix(3,0),
	//	pose.matrix(0,1),pose.matrix(1,1),pose.matrix(2,1),pose.matrix(3,1),
	//	pose.matrix(0,2),pose.matrix(1,2),pose.matrix(2,2),pose.matrix(3,2),
	//	pose.matrix(0,3),pose.matrix(1,3),pose.matrix(2,3),pose.matrix(3,3));
	/*Matrix4f M_d(affine.matrix(0,0),affine.matrix(1,0),affine.matrix(2,0),affine.matrix(3,0),
				 affine.matrix(0,1),affine.matrix(1,1),affine.matrix(2,1),affine.matrix(3,1),
				 affine.matrix(0,2),affine.matrix(1,2),affine.matrix(2,2),affine.matrix(3,2),
				 affine.matrix(0,3),affine.matrix(1,3),affine.matrix(2,3),affine.matrix(3,3));*/
	visualisationEngine->CreateExpectedDepths(scene,pose.inv(),p.intr,renderState);
    visualisationEngine->CreateICPMaps(scene,M_d,p.intr,prev_.points_pyr[0], prev_.normals_pyr[0],renderState);
    for (int i = 1; i < LEVELS; ++i)
        resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
    /*cv::Mat points_mat10(480,640,CV_32FC4);

	prev_.points_pyr[0].download(points_mat10.data,points_mat10.step); */
	/*cv::Mat points_mat(480,640,CV_32FC4);

	prev_.points_pyr[0].download(points_mat.data,points_mat.step); 
	cv::Mat points_mat1(240,320,CV_32FC4);

	prev_.points_pyr[1].download(points_mat1.data,points_mat1.step); 
	cv::Mat points_mat2(120,160,CV_32FC4);

	prev_.points_pyr[2].download(points_mat2.data,points_mat2.step); */
	/*prev_.points_pyr.swap(curr_.points_pyr);
    prev_.normals_pyr.swap(curr_.normals_pyr);*/
	/*cv::Mat points_mat5(480,640,CV_32FC4);

	prev_.points_pyr[0].download(points_mat5.data,points_mat5.step); */
	 //prev_.points_pyr[0].download(points_mat.data,points_mat.step);
    ++frame_counter_;
    return true;
}

void tfusion::TopFu::renderImage(cuda::image4u& image)//,const Affine3f& pose_, int flag)
{
    const TopFuParams& p = params_;

    image.create(p.rows,p.cols);
    //IVisualisationEngine::RenderImageType imageType = tfusion::IVisualisationEngine::RENDER_SHADED_GREYSCALE_IMAGENORMALS;
	IVisualisationEngine::RenderImageType imageType = tfusion::IVisualisationEngine::RENDER_SHADED_GREYSCALE;
    IVisualisationEngine::RenderRaycastSelection raycastType = tfusion::IVisualisationEngine::RENDER_FROM_NEW_RAYCAST;

    Affine3f pose = poses_.back();
    /*Matrix4f M_d(pose.matrix(0,0),pose.matrix(0,1),pose.matrix(0,2),pose.matrix(0,3),
                pose.matrix(1,0),pose.matrix(1,1),pose.matrix(1,2),pose.matrix(1,3),
                pose.matrix(2,0),pose.matrix(2,1),pose.matrix(2,2),pose.matrix(2,3),
                pose.matrix(3,0),pose.matrix(3,1),pose.matrix(3,2),pose.matrix(3,3));*/
	Matrix4f M_d(pose.matrix(0,0),pose.matrix(1,0),pose.matrix(2,0),pose.matrix(3,0),
		pose.matrix(0,1),pose.matrix(1,1),pose.matrix(2,1),pose.matrix(3,1),
		pose.matrix(0,2),pose.matrix(1,2),pose.matrix(2,2),pose.matrix(3,2),
		pose.matrix(0,3),pose.matrix(1,3),pose.matrix(2,3),pose.matrix(3,3));

    Vector4f intrs(p.intr.fx,p.intr.fy,p.intr.cx,p.intr.cy);
    // visualisationEngine->RenderImage(scene,M_d,intrs,renderState,renderState->raycastImage,imageType,raycastType);
    visualisationEngine->RenderImage(scene,M_d,intrs,renderState,image,imageType,raycastType);
    // Affine3f pose = pose_.inv();
    // // Vector2i imgSize = outputImage->noDims;
    // Vector2i imgSize(image.cols,image.rows);
    // // Matrix4f invM = pose->GetInvM();
    // Matrix4f M_d(pose.matrix(0,0),pose.matrix(0,1),pose.matrix(0,2),pose.matrix(0,3),
    //             pose.matrix(1,0),pose.matrix(1,1),pose.matrix(1,2),pose.matrix(1,3),
    //             pose.matrix(2,0),pose.matrix(2,1),pose.matrix(2,2),pose.matrix(2,3),
    //             pose.matrix(3,0),pose.matrix(3,1),pose.matrix(3,2),pose.matrix(3,3));

    // Vector4f *pointsRay;

    // GeneticRaycast(scene,imgSize,invM,intrinsics->projectionParamsSimple.all,renderState,false);
    // pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

    // Vector3f lightSource = -Vector3f(invM.getColumn(2));

    // Vector4u * outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);

    // dim3 cudaBlockSize(8,8);
    // dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x),(int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

    // rederGrey_ImageNormals_device<false><<<gridSize,cudaBlockSize>>>(outRendering,pointsRay,scene->sceneParams->voxelSize,imgSize,lightSource);
    
}


// void tfusion::TopFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
// {
//     const TopFuParams& p = params_;
//     image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
//     depths_.create(p.rows, p.cols);
//     normals_.create(p.rows, p.cols);
//     points_.create(p.rows, p.cols);

// #if defined USE_DEPTH
//     #define PASS1 depths_
// #else
//     #define PASS1 points_
// #endif

//     volume_->raycast(pose, p.intr, PASS1, normals_);

//     if (flag < 1 || flag > 3)
//         cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
//     else if (flag == 2)
//         cuda::renderTangentColors(normals_, image);
//     else /* if (flag == 3) */
//     {
//         DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
//         DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

//         cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
//         cuda::renderTangentColors(normals_, i2);
//     }
// #undef PASS1
// }