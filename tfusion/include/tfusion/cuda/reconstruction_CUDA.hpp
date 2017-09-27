#pragma once

#include "tfusion/cuda/SceneReconstructionEngine.hpp"
#include <tfusion/cuda/CUDAUtils.hpp>
#include <tfusion/Defines.hpp>
#include <tfusion/cuda/reconstruction.hpp>

namespace tfusion
{
	template<class TVoxel>
	class SceneReconstructionEngine_CUDA
	{
	private:
		void *allocationTempData_device;
		void *allocationTempData_host;
		unsigned char *entriesAllocType_device;
		Vector4s *blockCoords_device;

	public:
		void ResetScene(Scene<TVoxel, VoxelBlockHash> *scene);

		void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
		void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	const Affine3f pose, cuda::Dists& dist, const RenderState *renderState);

		SceneReconstructionEngine_CUDA(void);
		~SceneReconstructionEngine_CUDA(void);
	};

}
