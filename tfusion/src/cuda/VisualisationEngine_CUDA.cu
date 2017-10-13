#include "tfusion/cuda/VisualisationEngine_CUDA.hpp"
#include "tfusion/cuda/VisualisationHelper.hpp"
#include "tfusion/types.hpp"

namespace tfusion
{
inline dim3 getGridSize(dim3 taskSize, dim3 blockSize)
{
	return dim3((taskSize.x + blockSize.x - 1) / blockSize.x, (taskSize.y + blockSize.y - 1) / blockSize.y, (taskSize.z + blockSize.z - 1) / blockSize.z);
}

inline dim3 getGridSize(Vector2i taskSize, dim3 blockSize) { return getGridSize(dim3(taskSize.x, taskSize.y), blockSize); }

// template<class TVoxel, class TIndex>
// VisualisationEngine_CUDA<TVoxel, TIndex>::VisualisationEngine_CUDA(void)
// {
// 	ORcudaSafeCall(cudaMalloc((void**)&noTotalPoints_device, sizeof(uint)));
// }

// template<class TVoxel, class TIndex>
// VisualisationEngine_CUDA<TVoxel, TIndex>::~VisualisationEngine_CUDA(void)
// {
// 	ORcudaSafeCall(cudaFree(noTotalPoints_device));
// }

// template<class TVoxel, class TIndex>
// RenderState* VisualisationEngine_CUDA<TVoxel, TIndex>::CreateRenderState(const Scene<TVoxel, TIndex> *scene, const Vector2i & imgSize) const
// {
// 	return new RenderState(
// 		imgSize, scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max
// 	);
// }

// template<class TVoxel>
// RenderState_VH* VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::CreateRenderState(const Scene<TVoxel, VoxelBlockHash> *scene, const Vector2i & imgSize) const
// {
// 	return new RenderState_VH(
// 		VoxelBlockHash::noTotalEntries, imgSize, scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max, MEMORYDEVICE_CUDA
// 	);
// }

// template<class TVoxel>
// void VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::FindVisibleBlocks(const Scene<TVoxel,VoxelBlockHash> *scene, const Matrix4f pose, const Vector4f intrinsics, RenderState *renderState) const
// {
// 	const HashEntry *hashTable = scene->index.GetEntries();
// 	int noTotalEntries = scene->index.noTotalEntries;
// 	float voxelSize = scene->sceneParams->voxelSize;
// 	Vector2i imgSize = renderState->renderingRangeImage->noDims;

// 	// Matrix4f M = pose->GetM();
// 	Matrix4f M(pose);
// 	// Vector4f projParams = intrinsics->projectionParamsSimple.all;
// 	Vector4f projParams(intrinsics);

// 	RenderState_VH *renderState_vh = (RenderState_VH*)renderState;

// 	ORcudaSafeCall(cudaMemset(noVisibleEntries_device, 0, sizeof(int)));

// 	dim3 cudaBlockSizeAL(256, 1);
// 	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));
// 	buildCompleteVisibleList_device << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, /*cacheStates, this->scene->useSwapping,*/ noTotalEntries,
// 		renderState_vh->GetVisibleEntryIDs(), noVisibleEntries_device, renderState_vh->GetEntriesVisibleType(), M, projParams, 
// 		imgSize, voxelSize);
// 	ORcudaKernelCheck;

// 	/*	if (this->scene->useSwapping)
// 			{
// 			reAllocateSwappedOutVoxelBlocks_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, hashTable, noTotalEntries,
// 			noAllocatedVoxelEntries_device, entriesVisibleType);
// 			}*/

// 	ORcudaSafeCall(cudaMemcpy(&renderState_vh->noVisibleEntries, noVisibleEntries_device, sizeof(int), cudaMemcpyDeviceToHost));
// }

// template<class TVoxel, class TIndex>
// int VisualisationEngine_CUDA<TVoxel, TIndex>::CountVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const RenderState *renderState, int minBlockId, int maxBlockId) const
// {
// 	return 1;
// }

// template<class TVoxel>
// int VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::CountVisibleBlocks(const Scene<TVoxel,VoxelBlockHash> *scene, const RenderState *renderState, int minBlockId, int maxBlockId) const
// {
// 	const RenderState_VH *renderState_vh = (const RenderState_VH*)renderState;

// 	int noVisibleEntries = renderState_vh->noVisibleEntries;
// 	const int *visibleEntryIDs_device = renderState_vh->GetVisibleEntryIDs();

// 	ORcudaSafeCall(cudaMemset(noTotalBlocks_device, 0, sizeof(uint)));

// 	dim3 blockSize(256);
// 	dim3 gridSize((int)ceil((float)noVisibleEntries / (float)blockSize.x));

// 	const HashEntry *hashTable_device = scene->index.GetEntries();
// 	countVisibleBlocks_device<<<gridSize,blockSize>>>(visibleEntryIDs_device, noVisibleEntries, hashTable_device, noTotalBlocks_device, minBlockId, maxBlockId);
// 	ORcudaKernelCheck;

// 	uint noTotalBlocks;
// 	ORcudaSafeCall(cudaMemcpy(&noTotalBlocks, noTotalBlocks_device, sizeof(uint), cudaMemcpyDeviceToHost));

// 	return noTotalBlocks;
// }

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::CreateExpectedDepths(const Scene<TVoxel,TIndex> *scene, const Matrix4f pose,
// 	const Vector4f intrinsics, RenderState *renderState) const
// {
// 	// Vector2f *minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
// 	Vector2f *minmaxData = rederState->renderingRangeImage.ptr();

// 	Vector2f init;
// 	//TODO : this could be improved a bit...
// 	init.x = 0.2f; init.y = 3.0f;
// 	// memsetKernel<Vector2f>(minmaxData, init, renderState->renderingRangeImage->dataSize);
// 	memsetKernel<Vector2f>(minmaxData,init,rederState->renderingRangeImage.cols()*rederState->renderingRangeImage.rows());
// }

// template<class TVoxel>
template<class TVoxel, class TIndex>
void VisualisationEngine_CUDA<TVoxel, TIndex>::CreateExpectedDepths(const Scene<TVoxel,TIndex> *scene, const Affine3f pose_, const Intr intr_,
	RenderState *renderState) const
{
	float voxelSize = scene->sceneParams->voxelSize;

	// Vector2i imgSize = renderState->renderingRangeImage->noDims;
	Vector2i imgSize(renderState->renderingRangeImage.cols(),renderState->renderingRangeImage.rows());
	// Vector2f *minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
	Vector2f *minmaxData = renderState->renderingRangeImage.ptr();

	Vector2f init;
	init.x = FAR_AWAY; init.y = VERY_CLOSE;
	// memsetKernel<Vector2f>(minmaxData, init, renderState->renderingRangeImage->dataSize);
	memsetKernel<Vector2f>(minmaxData,init,renderState->renderingRangeImage.cols() * renderState->renderingRangeImage.rows());

	RenderState_VH* renderState_vh = (RenderState_VH*)renderState;

	Matrix4f pose(pose_.matrix(0,0),pose_.matrix(0,1),pose_.matrix(0,2),pose_.matrix(0,3),
		pose_.matrix(1,0),pose_.matrix(1,1),pose_.matrix(1,2),pose_.matrix(1,3),
		pose_.matrix(2,0),pose_.matrix(2,1),pose_.matrix(2,2),pose_.matrix(2,3),
		pose_.matrix(3,0),pose_.matrix(3,1),pose_.matrix(3,2),pose_.matrix(3,3));

	Vector4f intr(intr_.fx,intr_.fy,intr_.cx,intr_.cy);
	//go through list of visible 8x8x8 blocks
	{
		const HashEntry *hash_entries = scene->index.GetEntries();
		const int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
		int noVisibleEntries = renderState_vh->noVisibleEntries;
		if (noVisibleEntries == 0) return;

		dim3 blockSize(256);
		dim3 gridSize((int)ceil((float)noVisibleEntries / (float)blockSize.x));
		ORcudaSafeCall(cudaMemset(noTotalBlocks_device, 0, sizeof(uint)));
		// projectAndSplitBlocks_device << <gridSize, blockSize >> >(hash_entries, visibleEntryIDs, noVisibleEntries, pose,
		// 	intrinsics->projectionParamsSimple.all, imgSize, voxelSize, renderingBlockList_device, noTotalBlocks_device);
		projectAndSplitBlocks_device << <gridSize, blockSize >> >(hash_entries, visibleEntryIDs, noVisibleEntries, pose,
			intr, imgSize, voxelSize, renderingBlockList_device, noTotalBlocks_device);
		ORcudaKernelCheck;
	}

	uint noTotalBlocks;
	ORcudaSafeCall(cudaMemcpy(&noTotalBlocks, noTotalBlocks_device, sizeof(uint), cudaMemcpyDeviceToHost));
	if (noTotalBlocks == 0) return;
	if (noTotalBlocks > (unsigned)MAX_RENDERING_BLOCKS) noTotalBlocks = MAX_RENDERING_BLOCKS;

	// go through rendering blocks
	{
		// fill minmaxData
		dim3 blockSize(16, 16);
		dim3 gridSize((unsigned int)ceil((float)noTotalBlocks / 4.0f), 4);
		fillBlocks_device << <gridSize, blockSize >> >(noTotalBlocks, renderingBlockList_device, imgSize, minmaxData);
		ORcudaKernelCheck;
	}
}

template <class TVoxel, class TIndex>
static void GenericRaycast(const Scene<TVoxel, TIndex> *scene, const Vector2i& imgSize, const Matrix4f& invM, const Vector4f& projParams, RenderState *renderState, bool updateVisibleList)
{
	float voxelSize = scene->sceneParams->voxelSize;
	float oneOverVoxelSize = 1.0f / voxelSize;

	uchar *entriesVisibleType = NULL;
	if (updateVisibleList&&(dynamic_cast<const RenderState_VH*>(renderState)!=NULL))
	{
		entriesVisibleType = ((RenderState_VH*)renderState)->GetEntriesVisibleType();
	}

	dim3 cudaBlockSize(16, 12);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));
	if (entriesVisibleType!=NULL) genericRaycast_device<TVoxel, VoxelBlockHash, true> << <gridSize, cudaBlockSize >> >(
			// renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
			renderState->raycastResult.ptr(),
			entriesVisibleType,
			scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(),
			imgSize,
			invM,
			InvertProjectionParams(projParams),
			oneOverVoxelSize,
			// renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA),
			renderState->renderingRangeImage.ptr(),
			scene->sceneParams->mu
		);
	else genericRaycast_device<TVoxel, VoxelBlockHash, false> << <gridSize, cudaBlockSize >> >(
			// renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
			renderState->raycastResult.ptr(),
			NULL,
			scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(),
			imgSize,
			invM,
			InvertProjectionParams(projParams),
			oneOverVoxelSize,
			// renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA),
			renderState->renderingRangeImage.ptr(),
			scene->sceneParams->mu
		);
	ORcudaKernelCheck;
}

template<class TVoxel, class TIndex>
static void RenderImage_common(const tfusion::Scene<Voxel_s, VoxelBlockHash> *scene, Matrix4f pose,const Vector4f intrinsics,tfusion::RenderState *renderState,
	tfusion::cuda::image4u &outputImage, tfusion::IVisualisationEngine::RenderImageType type, tfusion::IVisualisationEngine::RenderRaycastSelection raycastType)
{
	// Vector2i imgSize = outputImage->noDims;
	Vector2i imgSize(outputImage.cols(),outputImage.rows());
	// Matrix4f invM = pose->GetInvM();
	Matrix4f invM;
	pose.inv(invM);

	// Vector4f *pointsRay;
	// if (raycastType == IVisualisationEngine::RENDER_FROM_OLD_RAYCAST) {
		// pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
		// pointsRay = renderState->raycastResult.ptr();
	// } else if (raycastType == IVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ) {
		// pointsRay = renderState->forwardProjection->GetData(MEMORYDEVICE_CUDA);
		// pointsRay = renderState->forwardProjection.ptr();
	// } else {
		GenericRaycast(scene, imgSize, invM, intrinsics, renderState, false);
		// pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
	Vector4f *pointsRay = renderState->raycastResult.ptr();
	// }

	Vector3f lightSource = -Vector3f(invM.getColumn(2));

	Vector4u *outRendering = outputImage.ptr();

	dim3 cudaBlockSize(8, 8);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

	if ((type == IVisualisationEngine::RENDER_COLOUR_FROM_VOLUME)&&
	    (!TVoxel::hasColorInformation)) type = IVisualisationEngine::RENDER_SHADED_GREYSCALE;

	switch (type) {
	case IVisualisationEngine::RENDER_COLOUR_FROM_VOLUME:
		renderColour_device<TVoxel, TIndex> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize);
		ORcudaKernelCheck;
		break;
	case IVisualisationEngine::RENDER_COLOUR_FROM_NORMAL:
		renderColourFromNormal_device<TVoxel, TIndex> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		ORcudaKernelCheck;
		break;
	case IVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE:
		renderColourFromConfidence_device<TVoxel, TIndex> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		ORcudaKernelCheck;
		break;
	case IVisualisationEngine::RENDER_SHADED_GREYSCALE_IMAGENORMALS:
		// if (intrinsics->FocalLengthSignsDiffer())
		if(false)
		{
			renderGrey_ImageNormals_device<true> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->sceneParams->voxelSize,
				imgSize, lightSource);
		}
		else
		{
			renderGrey_ImageNormals_device<false> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->sceneParams->voxelSize,
				imgSize, lightSource);
		}
		ORcudaKernelCheck;
		break;
	case IVisualisationEngine::RENDER_SHADED_GREYSCALE:
	default:
		renderGrey_device<TVoxel, TIndex> <<<gridSize, cudaBlockSize>>>(outRendering, pointsRay, scene->localVBA.GetVoxelBlocks(),
			scene->index.getIndexData(), imgSize, lightSource);
		ORcudaKernelCheck;
		break;
	}
}

// template<class TVoxel, class TIndex>
// static void CreatePointCloud_common(const Scene<TVoxel, TIndex> *scene, const Matrix4f pose, Vector4f calib,TrackingState *trackingState, RenderState *renderState,
// 	bool skipPoints, uint *noTotalPoints_device)
// {
// 	// Vector2i imgSize = renderState->raycastResult->noDims;
// 	Vector2i imgSize(renderState->raycastResult.cols(),renderState->raycastResult.rows());
// 	// Matrix4f invM = trackingState->pose_d->GetInvM() * view->calib.trafo_rgb_to_depth.calib;
// 	Matrix4f invM;
// 	pose.inv(invM);

// 	// GenericRaycast(scene, imgSize, invM, view->calib.intrinsics_rgb.projectionParamsSimple.all, renderState, true);
// 	GenericRaycast(scene,imgSize,invM,);
// 	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);

// 	ORcudaSafeCall(cudaMemsetAsync(noTotalPoints_device, 0, sizeof(uint)));

// 	Vector3f lightSource = -Vector3f(invM.getColumn(2));
// 	Vector4f *locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
// 	Vector4f *colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
// 	Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

// 	dim3 cudaBlockSize(16, 16);
// 	dim3 gridSize = getGridSize(imgSize, cudaBlockSize);
// 	renderPointCloud_device<TVoxel, TIndex> << <gridSize, cudaBlockSize >> >(locations, colours, noTotalPoints_device,
// 		pointsRay, scene->localVBA.GetVoxelBlocks(), scene->index.getIndexData(), skipPoints, scene->sceneParams->voxelSize, imgSize, lightSource);
// 	ORcudaKernelCheck;

// 	ORcudaSafeCall(cudaMemcpy(&trackingState->pointCloud->noTotalPoints, noTotalPoints_device, sizeof(uint), cudaMemcpyDeviceToHost));
// }

template<class TVoxel, class TIndex>
void CreateICPMaps_common(const Scene<TVoxel, TIndex> *scene, Matrix4f pose,Vector4f intr,Vector4f *points,Vector4f *normals, RenderState *renderState)
{
	// Vector2i imgSize = renderState->raycastResult->noDims;
	Vector2i imgSize(renderState->raycastResult.cols(),renderState->raycastResult.rows());
	// Matrix4f invM = trackingState->pose_d->GetInvM();
	Matrix4f invM = pose;

	GenericRaycast(scene, imgSize, invM, intr, renderState, true);
	// trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);

	// Vector4f *pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
	// Vector4f *normalsMap = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
	Vector4f* pointsMap = points;
	Vector4f* normalsMap = normals;
	// Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
	Vector4f *pointsRay = renderState->raycastResult.ptr();
	Vector3f lightSource = -Vector3f(invM.getColumn(2));

	dim3 cudaBlockSize(16, 12);
	dim3 gridSize((int)ceil((float)imgSize.x / (float)cudaBlockSize.x), (int)ceil((float)imgSize.y / (float)cudaBlockSize.y));

	// if (view->calib.intrinsics_d.FocalLengthSignsDiffer())
	if(false)
	{
		renderICP_device<true> <<<gridSize, cudaBlockSize>>>(pointsMap, normalsMap, pointsRay,
			scene->sceneParams->voxelSize, imgSize, lightSource);
	}
	else
	{
		renderICP_device<false> <<<gridSize, cudaBlockSize>>>(pointsMap, normalsMap, pointsRay,
			scene->sceneParams->voxelSize, imgSize, lightSource);
	}
	ORcudaKernelCheck;
}

// template<class TVoxel, class TIndex>
// static void ForwardRender_common(const Scene<TVoxel, TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState, 
// 	uint *noTotalPoints_device)
// {
// 	Vector2i imgSize = renderState->raycastResult->noDims;
// 	Matrix4f M = trackingState->pose_d->GetM();
// 	Matrix4f invM = trackingState->pose_d->GetInvM();
// 	const Vector4f& projParams = view->calib.intrinsics_d.projectionParamsSimple.all;

// 	const Vector4f *pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
// 	float *currentDepth = view->depth->GetData(MEMORYDEVICE_CUDA);
// 	Vector4f *forwardProjection = renderState->forwardProjection->GetData(MEMORYDEVICE_CUDA);
// 	int *fwdProjMissingPoints = renderState->fwdProjMissingPoints->GetData(MEMORYDEVICE_CUDA);
// 	const Vector2f *minmaximg = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
// 	float oneOverVoxelSize = 1.0f / scene->sceneParams->voxelSize;
// 	float voxelSize = scene->sceneParams->voxelSize;
// 	const TVoxel *voxelData = scene->localVBA.GetVoxelBlocks();
// 	const typename TIndex::IndexData *voxelIndex = scene->index.getIndexData();

// 	renderState->forwardProjection->Clear();

// 	dim3 blockSize, gridSize;

// 	{ // forward projection
// 		blockSize = dim3(16, 16);
// 		gridSize = dim3((int)ceil((float)imgSize.x / (float)blockSize.x), (int)ceil((float)imgSize.y / (float)blockSize.y));

// 		forwardProject_device << <gridSize, blockSize >> >(forwardProjection, pointsRay, imgSize, M, projParams, voxelSize);
// 		ORcudaKernelCheck;
// 	}

// 	ORcudaSafeCall(cudaMemset(noTotalPoints_device, 0, sizeof(uint)));

// 	{ // find missing points
// 		blockSize = dim3(16, 16);
// 		gridSize = dim3((int)ceil((float)imgSize.x / (float)blockSize.x), (int)ceil((float)imgSize.y / (float)blockSize.y));

// 		findMissingPoints_device << <gridSize, blockSize >> >(fwdProjMissingPoints, noTotalPoints_device, minmaximg, 
// 			forwardProjection, currentDepth, imgSize);
// 		ORcudaKernelCheck;
// 	}

// 	ORcudaSafeCall(cudaMemcpy(&renderState->noFwdProjMissingPoints, noTotalPoints_device, sizeof(uint), cudaMemcpyDeviceToHost));

// 	{ // render missing points
// 		blockSize = dim3(256);
// 		gridSize = dim3((int)ceil((float)renderState->noFwdProjMissingPoints / blockSize.x));

// 		genericRaycastMissingPoints_device<TVoxel, TIndex, false> << <gridSize, blockSize >> >(forwardProjection, NULL, voxelData, voxelIndex, imgSize, invM,
// 			InvertProjectionParams(projParams), oneOverVoxelSize, fwdProjMissingPoints, renderState->noFwdProjMissingPoints, minmaximg, scene->sceneParams->mu);
// 		ORcudaKernelCheck;
// 	}
// }

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::RenderImage(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, const RenderState *renderState,
// 	UChar4Image *outputImage, IVisualisationEngine::RenderImageType type,
// 	IVisualisationEngine::RenderRaycastSelection raycastType) const
// {
// 	RenderImage_common(scene, pose, intrinsics, renderState, outputImage, type, raycastType);
// }
template<class TVoxel, class TIndex>
void VisualisationEngine_CUDA<TVoxel, TIndex>::RenderImage(const Scene<TVoxel,TIndex> *scene, Matrix4f pose, const Vector4f intrinsics, RenderState *renderState,
	cuda::image4u& outputImage, IVisualisationEngine::RenderImageType type,
	IVisualisationEngine::RenderRaycastSelection raycastType) const
{
	RenderImage_common<TVoxel,TIndex>(scene, pose, intrinsics, renderState, outputImage, type, raycastType);
}

// template<class TVoxel>
// void VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::RenderImage(const Scene<TVoxel,VoxelBlockHash> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics,
// 	const RenderState *renderState, UChar4Image *outputImage, IVisualisationEngine::RenderImageType type,
// 	IVisualisationEngine::RenderRaycastSelection raycastType) const
// {
// 	RenderImage_common(scene, pose, intrinsics, renderState, outputImage, type, raycastType);
// }

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::FindSurface(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, const RenderState *renderState) const
// {
// 	GenericRaycast(scene, renderState->raycastResult->noDims, pose->GetInvM(), intrinsics->projectionParamsSimple.all, renderState, false);
// }

// template<class TVoxel>
// void VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::FindSurface(const Scene<TVoxel,VoxelBlockHash> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics,
// 	const RenderState *renderState) const
// {
// 	GenericRaycast(scene, renderState->raycastResult->noDims, pose->GetInvM(), intrinsics->projectionParamsSimple.all, renderState, false);
// }

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::CreatePointCloud(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState, 
// 	bool skipPoints) const
// {
// 	CreatePointCloud_common(scene, view, trackingState, renderState, skipPoints, noTotalPoints_device);
// }

// template<class TVoxel>
// void VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::CreatePointCloud(const Scene<TVoxel,VoxelBlockHash> *scene,const View *view, TrackingState *trackingState, 
// 	RenderState *renderState, bool skipPoints) const
// {
// 	CreatePointCloud_common(scene, view, trackingState, renderState, skipPoints, noTotalPoints_device);
// }

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::CreateICPMaps(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, 
// 	RenderState *renderState) const
// {
// 	CreateICPMaps_common(scene, view, trackingState, renderState);
// }

template<class TVoxel,class TIndex>
void VisualisationEngine_CUDA<TVoxel, TIndex>::CreateICPMaps(const Scene<TVoxel,TIndex> *scene, const Affine3f pose_,const Intr intr_,cuda::Cloud &points_,cuda::Normals &normals_, 
	RenderState *renderState) const
{

	Affine3f pose_inv = pose_.inv();
	Matrix4f pose(pose_inv.matrix(0,0),pose_inv.matrix(0,1),pose_inv.matrix(0,2),pose_inv.matrix(0,3),
		pose_inv.matrix(1,0),pose_inv.matrix(1,1),pose_inv.matrix(1,2),pose_inv.matrix(1,3),
		pose_inv.matrix(2,0),pose_inv.matrix(2,1),pose_inv.matrix(2,2),pose_inv.matrix(2,3),
		pose_inv.matrix(3,0),pose_inv.matrix(3,1),pose_inv.matrix(3,2),pose_inv.matrix(3,3));

	Vector4f intr(intr_.fx,intr_.fy,intr_.cx,intr_.cy);

	Vector4f *points = (Vector4f*)(points_.ptr());
	Vector4f *normals = (Vector4f*)(normals_.ptr());


	// CreateICPMaps_common(scene, view, trackingState, renderState);
	CreateICPMaps_common(scene,pose,intr,points,normals,renderState);
}

// template<class TVoxel, class TIndex>
// void VisualisationEngine_CUDA<TVoxel, TIndex>::ForwardRender(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, 
// 	RenderState *renderState) const
// {
// 	ForwardRender_common(scene, view, trackingState, renderState, this->noTotalPoints_device);
// }

// template<class TVoxel>
// void VisualisationEngine_CUDA<TVoxel, VoxelBlockHash>::ForwardRender(const Scene<TVoxel,VoxelBlockHash> *scene, const View *view, TrackingState *trackingState, 
// 	RenderState *renderState) const
// {
// 	ForwardRender_common(scene, view, trackingState, renderState, this->noTotalPoints_device);
// }
}