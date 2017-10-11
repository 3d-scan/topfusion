#pragma once

// #include "tfusion/cuda/SceneReconstructionEngine.hpp"
// #include <tfusion/cuda/CUDAUtils.hpp>
#include <tfusion/Defines.hpp>
// #include <tfusion/cuda/reconstruction.hpp>
#include <tfusion/scene.hpp>
#include <tfusion/RenderState.hpp>
#include <tfusion/cuda/SceneReconstruction.hpp>
// #include <tfusion/cuda/SceneReconstructionEngine_CUDA.hpp>

struct AllocationTempData{

	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries; 
};

namespace tfusion
{
	// template<class TVoxel, class TIndex>
	// class SceneReconstructionEngine_CUDA : public SceneReconstructionEngine < TVoxel, TIndex >
	// {
	// private:
	// 	void *allocationTempData_device;
	// 	void *allocationTempData_host;
	// 	unsigned char *entriesAllocType_device;
	// 	Vector4s *blockCoords_device;

	// public:
	// 	void ResetScene(Scene<TVoxel, VoxelBlockHash> *scene);

	// 	void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	// const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
	// 	void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	// const Affine3f pose, cuda::Dists& dist, const RenderState *renderState);

	// 	SceneReconstructionEngine_CUDA(void);
	// 	~SceneReconstructionEngine_CUDA(void);
	// };

	template<class TVoxel,class TIndex>
	class SceneReconstructionEngine_CUDA//<TVoxel,VoxelBlockHash> : public SceneReconstructionEngine<TVoxel,VoxelBlockHash>
	{
	private:
		void *allocationTempData_device;
		void *allocationTempData_host;
		unsigned char *entriesAllocType_device;
		Vector4s *blockCoords_device;

	public:
		void ResetScene(Scene<Voxel_s, VoxelBlockHash> *scene);

		void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
		void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	const Affine3f pose, cuda::Dists& dist, const RenderState *renderState);

		SceneReconstructionEngine_CUDA(void){
			ORcudaSafeCall(cudaMalloc((void**)&allocationTempData_device, sizeof(AllocationTempData)));
			ORcudaSafeCall(cudaMallocHost((void**)&allocationTempData_host, sizeof(AllocationTempData)));

			int noTotalEntries = VoxelBlockHash::noTotalEntries;
			ORcudaSafeCall(cudaMalloc((void**)&entriesAllocType_device, noTotalEntries));
			ORcudaSafeCall(cudaMalloc((void**)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));
		}
		~SceneReconstructionEngine_CUDA(void){
			ORcudaSafeCall(cudaFreeHost(allocationTempData_host));
			ORcudaSafeCall(cudaFree(allocationTempData_device));
			ORcudaSafeCall(cudaFree(entriesAllocType_device));
			ORcudaSafeCall(cudaFree(blockCoords_device));
		}
		// void hello();
	};

	// class SceneReconstructionEngine_CUDA
	// {
	// private:
	// 	void *allocationTempData_device;
	// 	void *allocationTempData_host;
	// 	unsigned char *entriesAllocType_device;
	// 	Vector4s *blockCoords_device;

	// public:
	// 	void ResetScene(Scene<TVoxel, VoxelBlockHash> *scene);

	// 	void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	// const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
	// 	void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	// const Affine3f pose, cuda::Dists& dist, const RenderState *renderState);

	// 	SceneReconstructionEngine_CUDA(void);
	// 	~SceneReconstructionEngine_CUDA(void);
	// 	// void hello();
	// };

	// void hello();
}