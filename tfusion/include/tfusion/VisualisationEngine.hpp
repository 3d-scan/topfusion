#pragma once

#include <tfusion/RenderState.hpp>
#include <tfusion/RenderState_VH.hpp>
#include <tfusion/scene.hpp>
#include <tfusion/types.hpp>
// #include "../../../Objects/Tracking/TrackingState.h"
// #include "../../../Objects/Views/View.h"

namespace tfusion
{
	class IVisualisationEngine
	{
	public:
		enum RenderImageType
		{
			RENDER_SHADED_GREYSCALE,
			RENDER_SHADED_GREYSCALE_IMAGENORMALS,
			RENDER_COLOUR_FROM_VOLUME,
			RENDER_COLOUR_FROM_NORMAL,
			RENDER_COLOUR_FROM_CONFIDENCE
		};

		enum RenderRaycastSelection
		{
			RENDER_FROM_NEW_RAYCAST,
			RENDER_FROM_OLD_RAYCAST,
			RENDER_FROM_OLD_FORWARDPROJ
		};

		virtual ~IVisualisationEngine(void) {}

		// static void DepthToUchar4(UChar4Image *dst, const FloatImage *src);
		// static void NormalToUchar4(UChar4Image* dst, const Float4Image *src);
		// static void WeightToUchar4(UChar4Image *dst, const FloatImage *src);
	};

	template<class TIndex> struct IndexToRenderState { typedef RenderState type; };
	template<> struct IndexToRenderState<VoxelBlockHash> { typedef RenderState_VH type; };

	/** \brief
		Interface to engines helping with the visualisation of
		the results from the rest of the library.

		This is also used internally to get depth estimates for the
		raycasting done for the trackers. The basic idea there is
		to project down a scene of 8x8x8 voxel
		blocks and look at the bounding boxes. The projection
		provides an idea of the possible depth range for each pixel
		in an image, which can be used to speed up raycasting
		operations.
		*/
	template<class TVoxel, class TIndex>
	class VisualisationEngine : public IVisualisationEngine
	{
	public:
		/** Creates a render state, containing rendering info
		for the scene.
		*/
		// virtual typename IndexToRenderState<TIndex>::type *CreateRenderState(const Scene<TVoxel, TIndex> *scene, const Vector2i & imgSize) const = 0;

		/** Given a scene, pose and intrinsics, compute the
		visible subset of the scene and store it in an
		appropriate visualisation state object, created
		previously using allocateInternalState().
		*/
		// virtual void FindVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics,
		// 	RenderState *renderState) const = 0;

		/** Given a render state, Count the number of visible blocks
		with minBlockId <= blockID <= maxBlockId .
		*/
		// virtual int CountVisibleBlocks(const Scene<TVoxel,TIndex> *scene, const RenderState *renderState, int minBlockId = 0, int maxBlockId = SDF_LOCAL_BLOCK_NUM) const = 0;

		/** Given scene, pose and intrinsics, create an estimate
		of the minimum and maximum depths at each pixel of
		an image.
		*/
		virtual void CreateExpectedDepths(const Scene<TVoxel,TIndex> *scene, const Affine3f pose, const Intr intrinsics, RenderState *renderState) const = 0;

		/** This will render an image using raycasting. */
		virtual void RenderImage(const Scene<TVoxel,TIndex> *scene, Matrix4f pose, const Vector4f intrinsics, RenderState *renderState,
	cuda::image4u& outputImage, RenderImageType type = RENDER_SHADED_GREYSCALE, RenderRaycastSelection raycastType = RENDER_FROM_NEW_RAYCAST) const = 0;

		/** Finds the scene surface using raycasting. */
		// virtual void FindSurface(const Scene<TVoxel,TIndex> *scene, const ORUtils::SE3Pose *pose, const Intrinsics *intrinsics,
		// 	const RenderState *renderState) const = 0;

		/** Create a point cloud as required by the
		Lib::Engine::ColorTracker classes.
		*/
		// virtual void CreatePointCloud(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState, 
		// 	RenderState *renderState, bool skipPoints) const = 0;

		/** Create an image of reference points and normals as
		required by the Lib::Engine::DepthTracker classes.
		*/
		virtual void CreateICPMaps(const Scene<TVoxel,VoxelBlockHash> *scene, const Matrix4f pose_,const Intr intr_,cuda::Cloud &points_,cuda::Normals &normals_, 
	RenderState *renderState) const = 0;

		/** Create an image of reference points and normals as
		required by the Lib::Engine::DepthTracker classes.

		Incrementally previous raycast result.
		*/
		// virtual void ForwardRender(const Scene<TVoxel,TIndex> *scene, const View *view, TrackingState *trackingState,
			// RenderState *renderState) const = 0;
	};
}