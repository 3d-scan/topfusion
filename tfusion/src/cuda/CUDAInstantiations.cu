#include "SceneReconstructionEngine_host.cu"
#include "VisualisationEngine_CUDA.cu"

namespace tfusion{
	// template class ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
	// template class ITMMultiMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
	template class SceneReconstructionEngine_CUDA<Voxel_s, VoxelBlockHash>;
	// template class ITMSwappingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
	template class VisualisationEngine_CUDA<Voxel_s, VoxelBlockHash>;
	// template class ITMMultiVisualisationEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
}