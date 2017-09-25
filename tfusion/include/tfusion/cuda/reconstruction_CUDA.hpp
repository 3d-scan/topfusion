#pragma once

#include "SceneReconstructionEngine.h"

namespace tfusion
{
	template<class TVoxel>
	class SceneReconstructionEngine_CUDA<TVoxel, VoxelBlockHash> : public SceneReconstructionEngine < TVoxel, VoxelBlockHash >
	{
	private:
		void *allocationTempData_device;
		void *allocationTempData_host;
		unsigned char *entriesAllocType_device;
		Vector4s *blockCoords_device;

	public:
		void ResetScene(Scene<TVoxel, VoxelBlockHash> *scene);

		void AllocateSceneFromDepth(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr, 
	const Matrix4f pose, cuda::Dist &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false);
		void IntegrateIntoScene(Scene<TVoxel, VoxelBlockHash> *scene, const Intr intr,
	const Matrix4f pose, cuda::Dist& dist, const RenderState *renderState);

		SceneReconstructionEngine_CUDA(void);
		~SceneReconstructionEngine_CUDA(void);
	};

}
