#pragma once

#include "tfusion/VisualisationEngine.hpp"
#include "tfusion/Defines.hpp"

struct RenderingBlock;

namespace tfusion
{
	template<class TVoxel, class TIndex>
	class VisualisationEngine_CUDA : public VisualisationEngine < TVoxel, TIndex >
	{
	private:
		uint *noTotalPoints_device;

	public:
		explicit VisualisationEngine_CUDA(void){
			ORcudaSafeCall(cudaMalloc((void**)&noTotalPoints_device, sizeof(uint)));
		}
		~VisualisationEngine_CUDA(void){
			ORcudaSafeCall(cudaFree(noTotalPoints_device));
		}

		// RenderState* CreateRenderState(const Scene<TVoxel, TIndex> *scene, const Vector2i & imgSize) const;
		// void FindVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, RenderState *renderState) const;
		// int CountVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const RenderState *renderState, int minBlockId, int maxBlockId) const;
		// void CreateExpectedDepths(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, RenderState *renderState) const;
		void RenderImage(const Scene<TVoxel,TIndex> *scene, Matrix4f pose, const Vector4f intrinsics, RenderState *renderState,
	cuda::image4u& outputImage, IVisualisationEngine::RenderImageType type = IVisualisationEngine::RENDER_SHADED_GREYSCALE,
			IVisualisationEngine::RenderRaycastSelection raycastType = IVisualisationEngine::RENDER_FROM_NEW_RAYCAST) const;
		// void FindSurface(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, const RenderState *renderState) const;
		// void CreatePointCloud(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState, bool skipPoints) const;
		// void CreateICPMaps(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState) const;
		// void ForwardRender(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState) const;
	};
}
