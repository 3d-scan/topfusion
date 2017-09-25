#pragma once

#include "VisualisationEngine.h"

struct RenderingBlock;

namespace tfusion
{
	template<class TVoxel, class TIndex>
	class VisualisationEngine_CUDA : public VisualisationEngine < TVoxel, TIndex >
	{
	private:
		uint *noTotalPoints_device;

	public:
		explicit VisualisationEngine_CUDA(void);
		~VisualisationEngine_CUDA(void);

		RenderState* CreateRenderState(const Scene<TVoxel, TIndex> *scene, const Vector2i & imgSize) const;
		void FindVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, RenderState *renderState) const;
		int CountVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const RenderState *renderState, int minBlockId, int maxBlockId) const;
		void CreateExpectedDepths(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, RenderState *renderState) const;
		void RenderImage(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, const RenderState *renderState,
			UChar4Image *outputImage, IVisualisationEngine::RenderImageType type = IVisualisationEngine::RENDER_SHADED_GREYSCALE,
			IVisualisationEngine::RenderRaycastSelection raycastType = IVisualisationEngine::RENDER_FROM_NEW_RAYCAST) const;
		void FindSurface(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics, const RenderState *renderState) const;
		void CreatePointCloud(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState, bool skipPoints) const;
		void CreateICPMaps(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState) const;
		void ForwardRender(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, RenderState *renderState) const;
	};
}
