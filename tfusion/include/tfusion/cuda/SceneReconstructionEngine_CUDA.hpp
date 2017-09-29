#pragma once

// // #include "tfusion/cuda/SceneReconstructionEngine.hpp"
// #include <tfusion/cuda/CUDAUtils.hpp>
#include <tfusion/Defines.hpp>
// // #include <tfusion/cuda/reconstruction.hpp>
// #include <tfusion/scene.hpp>
// #include <tfusion/RenderState.hpp>
#include <tfusion/GlobalCache.hpp>
struct AllocationTempData{

	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries; 
};
// namespace tfusion
// {
// 	template<class TVoxel>
// 	class SceneReconstructionEngine_CUDA
// 	{
// 	private:
// 		void *allocationTempData_device;
// 		void *allocationTempData_host;
// 		unsigned char *entriesAllocType_device;
// 		Vector4s *blockCoords_device;

// 	public:
// 		void ResetScene(Scene<TVoxel, VoxelBlockHash> *scene);

// 		void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
// 	const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
// 		void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
// 	const Affine3f pose, cuda::Dists& dist, const RenderState *renderState);

// 		SceneReconstructionEngine_CUDA(void);
// 		~SceneReconstructionEngine_CUDA(void);
// 	};

// }
using namespace tfusion;

// namespace 
// {

	template<class TVoxel,bool stopMaxW>
	__global__ void integrateIntoScene_device(TVoxel *localVBA, const HashEntry *hashTable, int *visibleEntryIDs,
		const ushort* depth, Vector2i depthImgSize, Matrix4f M_d, Vector4f projParams_d, 
		float _voxelSize, float mu, int maxW);

	// template<class TVoxel, bool stopMaxW>
	// __global__ void integrateIntoScene_device(TVoxel *voxelArray, const PlainVoxelArray::VoxelArrayInfo *arrayInfo,
	// 	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, const float *confidence, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	// 	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

	__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const ushort* depth,
		Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, HashEntry *hashTable, float viewFrustum_min,
		float viewFrustum_max);

	__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList, int *excessAllocationList, HashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData, uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords);

	__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, HashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData, uchar *entriesVisibleType);

	__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries);

	template<bool useSwapping>
	__global__ void buildVisibleList_device(HashEntry *hashTable, HashSwapState *swapStates, int noTotalEntries,
		int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType,
		Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize);
// }
