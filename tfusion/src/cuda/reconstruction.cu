#include <tfusion/cuda/reconstruction_CUDA.hpp>

struct AllocationTempData{

	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries; 
};

using namespace tfusion;

namespace{
	__global__ void integrateIntoScene_device(TVoxel *localVBA, const HashEntry *hashTable, int *noVisibleEntryIDs,
		const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, const float *confidence, Vector2i imgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
		Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

	template<class TVoxel, bool stopMaxW>
	__global__ void integrateIntoScene_device(TVoxel *voxelArray, const PlainVoxelArray::VoxelArrayInfo *arrayInfo,
		const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, const float *confidence, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
		Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

	__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
		Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, HashEntry *hashTable, float viewFrustum_min,
		float viewFrustrum_max);

	__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList, int *excessAllocationList, HashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData, uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords);

	__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, HashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData, uchar *entriesVisibleType);

	__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries);

	template<bool useSwapping>
	__global__ void buildVisibleList_device(HashEntry *hashTable, HashSwapState *swapStates, int noTotalEntries,
		int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType,
		Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize);
}

//host method

template<class TVoxel>
SceneReconstruction<TVoxel,VoxelBlockHash>::SceneReconstruction(void)
{
	ORcudaSafeCall(cudaMalloc((void**)&allocationTempData_device, sizeof(AllocationTempData)));
	ORcudaSafeCall(cudaMallocHost((void**)&allocationTempData_host, sizeof(AllocationTempData)));

	int noTotalEntries = VoxelBlockHash::noTotalEntries;
	ORcudaSafeCall(cudaMalloc((void**)&entriesAllocType_device, noTotalEntries));
	ORcudaSafeCall(cudaMalloc((void**)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));
}

template<class TVoxel>
SceneReconstruction<TVoxel,VoxelBlockHash>::~SceneReconstruction(void)
{
	ORcudaSafeCall(cudaFreeHost(allocationTempData_host));
	ORcudaSafeCall(cudaFree(allocationTempData_device));
	ORcudaSafeCall(cudaFree(entriesAllocType_device));
	ORcudaSafeCall(cudaFree(blockCoords_device));
}

template<class TVoxel>
void SceneReconstruction<TVoxel,VoxelBlockHash>::ResetScene(Scene<TVoxel, VoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

	HashEntry tmpEntry;
	memset(&tmpEntry, 0, sizeof(HashEntry));
	tmpEntry.ptr = -2;
	HashEntry *hashEntry_ptr = scene->index.GetEntries();
	memsetKernel<HashEntry>(hashEntry_ptr, tmpEntry, scene->index.noTotalEntries);
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	fillArrayKernel<int>(excessList_ptr, SDF_EXCESS_LIST_SIZE);

	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}
//modified by chuan
template<class TVoxel>
void SceneReconstruction<TVoxel, VoxelBlockHash>::AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	const Matrix4f pose, cuda::Dist &dist,const RenderState *renderState,bool onlyUpdateVisibleList, bool resetVisibleList)
{
	// Vector2i depthImgSize = view->depth->noDims;
	Vector2i depthImgSize(dist.cols,dist.rows);
	float voxelSize = scene->sceneParams->voxelSize;

	// Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	RenderState_VH *renderState_vh = (RenderState_VH*)renderState;

	if (resetVisibleList) renderState_vh->noVisibleEntries = 0;

	// Matrix4f M_d(pose.matrix(0,0),pose.matrix(0,1),pose.matrix(0,2),pose.matrix(0,3),
	// 			pose.matrix(1,0),pose.matrix(1,1),pose.matrix(1,2),pose.matrix(1,3),
	// 			pose.matrix(2,0),pose.matrix(2,1),pose.matrix(2,2),pose.matrix(2,3),
	// 			pose.matrix(3,0),pose.matrix(3,1),pose.matrix(3,2),pose.matrix(3,3));
	// M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);
	Matrix4f M_d(pose);
	Matrix4f invM_d;
	M_d.inv(invM_d);
	
	projParams_d = new Vector4f(intr.fx,intr.fy,intr.cx,intr.cy);
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	// float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	// ushort *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	HashEntry *hashTable = scene->index.GetEntries();
	HashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(true) : 0;

	int noTotalEntries = scene->index.noTotalEntries;

	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

	dim3 cudaBlockSizeHV(16, 16);
	dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

	dim3 cudaBlockSizeVS(256, 1);
	dim3 gridSizeVS((int)ceil((float)renderState_vh->noVisibleEntries / (float)cudaBlockSizeVS.x));

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	AllocationTempData *tempData = (AllocationTempData*)allocationTempData_host;
	tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
	tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
	tempData->noVisibleEntries = 0;
	ORcudaSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

	ORcudaSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* noTotalEntries));

	if (gridSizeVS.x > 0)
	{
		setToType3 << <gridSizeVS, cudaBlockSizeVS >> > (entriesVisibleType, visibleEntryIDs, renderState_vh->noVisibleEntries);
		ORcudaKernelCheck;
	}

	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(entriesAllocType_device, entriesVisibleType, 
		blockCoords_device, dist, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max);
	ORcudaKernelCheck;

	bool useSwapping = scene->globalCache != NULL;
	if (onlyUpdateVisibleList) useSwapping = false;
	//execute
	if (!onlyUpdateVisibleList)
	{
		allocateVoxelBlocksList_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, excessAllocationList, hashTable,
			noTotalEntries, (AllocationTempData*)allocationTempData_device, entriesAllocType_device, entriesVisibleType,
			blockCoords_device);
		ORcudaKernelCheck;
	}
	//no execute
	if (useSwapping)
	{
		buildVisibleList_device<true> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
		ORcudaKernelCheck;
	}
	//execute
	else
	{
		buildVisibleList_device<false> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
		ORcudaKernelCheck;
	}
	//no execute
	if (useSwapping)
	{
		reAllocateSwappedOutVoxelBlocks_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, hashTable, noTotalEntries, 
			(AllocationTempData*)allocationTempData_device, entriesVisibleType);
		ORcudaKernelCheck;
	}

	ORcudaSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
	renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
	scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
	scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);
}

template<class TVoxel>
void SceneReconstruction<TVoxel, VoxelBlockHash>::IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	const Matrix4f pose, cuda::Dist& dist, const RenderState *renderState)
{
	Vector2i depthImgSize(dist.cols,dist.rows);
	float voxelSize = scene->sceneParams->voxelSize;

	RenderState_VH *renderState_vh = (RenderState_VH*)renderState;
	if(renderState_vh->noVisibleEntries == 0) return;

	// M_d = trackingState->pose_d->GetM();
	// Matrix4f M_d(pose.matrix(0,0),pose.matrix(0,1),pose.matrix(0,2),pose.matrix(0,3),
	// 			pose.matrix(1,0),pose.matrix(1,1),pose.matrix(1,2),pose.matrix(1,3),
	// 			pose.matrix(2,0),pose.matrix(2,1),pose.matrix(2,2),pose.matrix(2,3),
	// 			pose.matrix(3,0),pose.matrix(3,1),pose.matrix(3,2),pose.matrix(3,3));
	Matrix4f M_d(pose);
	// if (TVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	Vector4f projParams_d(intr.fx,intr.fy,intr.cx,intr.cy);

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	// float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	// float *confidence = view->depthConfidence->GetData(MEMORYDEVICE_CUDA);
	// Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	HashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(renderState_vh->noVisibleEntries);

	if (scene->sceneParams->stopIntegratingAtMaxW)
	{
		// integrateIntoScene_device<TVoxel, true> << <gridSize, cudaBlockSize >> >(localVBA, hashTable, visibleEntryIDs,
		// 	rgb, rgbImgSize, depth, confidence, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		// ORcudaKernelCheck;
		integrateIntoScene_device<TVoxel, true> << <gridSize, cudaBlockSize >> >(localVBA, hashTable, visibleEntryIDs,(PtrStepSz<ushort>)dist, depthImgSize, M_d, projParams_d, voxelSize, mu, maxW);
		ORcudaKernelCheck;
	}
	//execute
	else
	{
		// integrateIntoScene_device<TVoxel, false> << <gridSize, cudaBlockSize >> >(localVBA, hashTable, visibleEntryIDs,
		// 	rgb, rgbImgSize, depth, confidence, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		// ORcudaKernelCheck;
		integrateIntoScene_device<TVoxel, false> << <gridSize, cudaBlockSize >> >(localVBA, hashTable, visibleEntryIDs,(PtrStepSz<ushort>)dist, depthImgSize, M_d, projParams_d, voxelSize, mu, maxW);
		ORcudaKernelCheck;
	}
}

namespace
{
	//device functions

template<class TVoxel, bool stopMaxW>
__global__ void integrateIntoScene_device(TVoxel *localVBA, const HashEntry *hashTable, int *visibleEntryIDs,
	const PtrStepSz<ushort> depth, Vector2i depthImgSize, Matrix4f M_d, Vector4f projParams_d, 
	float _voxelSize, float mu, int maxW)
{
	Vector3i globalPos;
	int entryId = visibleEntryIDs[blockIdx.x];

	const HashEntry &currentHashEntry = hashTable[entryId];

	if (currentHashEntry.ptr < 0) return;

	globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;

	TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;

	Vector4f pt_model; int locId;

	locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	if (stopMaxW) if (localVoxelBlock[locId].w_depth == maxW) return;
	//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) return;

	pt_model.x = (float)(globalPos.x + x) * _voxelSize;
	pt_model.y = (float)(globalPos.y + y) * _voxelSize;
	pt_model.z = (float)(globalPos.z + z) * _voxelSize;
	pt_model.w = 1.0f;

	ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation, TVoxel::hasConfidenceInformation, TVoxel>::compute(localVoxelBlock[locId], 
		pt_model, M_d, projParams_d, mu, maxW, depth, depthImgSize);
}

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const cuda::Dist &depth,
	Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, HashEntry *hashTable, float viewFrustum_min,
	float viewFrustum_max)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > _imgSize.x - 1 || y > _imgSize.y - 1) return;

	buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, (tfusion::cuda::PtrStepSz<ushort>)depth, invM_d,
		projParams_d, mu, _imgSize, _voxelSize, hashTable, viewFrustum_min, viewFrustum_max);
}

__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noVisibleEntries - 1) return;
	entriesVisibleType[visibleEntryIDs[entryId]] = 3;
}

__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList, int *excessAllocationList, HashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords)
{
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	int vbaIdx, exlIdx;

	switch (entriesAllocType[targetIdx])
	{
	case 1: //needs allocation, fits in the ordered list
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);

		if (vbaIdx >= 0) //there is room in the voxel block array
		{
			Vector4s pt_block_all = blockCoords[targetIdx];

			HashEntry hashEntry;
			hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
			hashEntry.ptr = voxelAllocationList[vbaIdx];
			hashEntry.offset = 0;

			hashTable[targetIdx] = hashEntry;
		}
		else
		{
			// Mark entry as not visible since we couldn't allocate it but buildHashAllocAndVisibleTypePP changed its state.
			entriesVisibleType[targetIdx] = 0;

			// Restore the previous value to avoid leaks.
			atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
		}
		break;

	case 2: //needs allocation in the excess list
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
		exlIdx = atomicSub(&allocData->noAllocatedExcessEntries, 1);

		if (vbaIdx >= 0 && exlIdx >= 0) //there is room in the voxel block array and excess list
		{
			Vector4s pt_block_all = blockCoords[targetIdx];

			HashEntry hashEntry;
			hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
			hashEntry.ptr = voxelAllocationList[vbaIdx];
			hashEntry.offset = 0;

			int exlOffset = excessAllocationList[exlIdx];

			hashTable[targetIdx].offset = exlOffset + 1; //connect to child

			hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

			entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; //make child visible
		}
		else
		{
			// No need to mark the entry as not visible since buildHashAllocAndVisibleTypePP did not mark it.
			// Restore the previous values to avoid leaks.
			atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
			atomicAdd(&allocData->noAllocatedExcessEntries, 1);
		}

		break;
	}
}

__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, HashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, /*int *noAllocatedVoxelEntries,*/ uchar *entriesVisibleType)
{
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	int vbaIdx;
	int hashEntry_ptr = hashTable[targetIdx].ptr;

	if (entriesVisibleType[targetIdx] > 0 && hashEntry_ptr == -1) //it is visible and has been previously allocated inside the hash, but deallocated from VBA
	{
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
		if (vbaIdx >= 0) hashTable[targetIdx].ptr = voxelAllocationList[vbaIdx];
		else atomicAdd(&allocData->noAllocatedVoxelEntries, 1);
	}
}

template<bool useSwapping>
__global__ void buildVisibleList_device(HashEntry *hashTable, HashSwapState *swapStates, int noTotalEntries,
	int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType, 
	Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize)
{
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	__shared__ bool shouldPrefix;
	shouldPrefix = false;
	__syncthreads();

	unsigned char hashVisibleType = entriesVisibleType[targetIdx];
	const HashEntry & hashEntry = hashTable[targetIdx];

	if (hashVisibleType == 3)
	{
		bool isVisibleEnlarged, isVisible;

		if (useSwapping)
		{
			checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
			if (!isVisibleEnlarged) hashVisibleType = 0;
		} else {
			checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
			if (!isVisible) hashVisibleType = 0;
		}
		entriesVisibleType[targetIdx] = hashVisibleType;
	}

	if (hashVisibleType > 0) shouldPrefix = true;

	if (useSwapping)
	{
		if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
	}

	__syncthreads();

	if (shouldPrefix)
	{
		int offset = computePrefixSum_device<int>(hashVisibleType > 0, &allocData->noVisibleEntries, blockDim.x * blockDim.y, threadIdx.x);
		if (offset != -1) visibleEntryIDs[offset] = targetIdx;
	}

}

}
