#pragma once

#include <math.h>

#include "tfusion/RenderState.hpp"
#include "tfusion/scene.hpp"

namespace tfusion
{
	/** \brief
	    Interface to engines implementing the main KinectFusion
	    depth integration process.

	    These classes basically manage
	    an ITMLib::Objects::ITMScene and fuse new image information
	    into them.
	*/
	template<class TVoxel, class TIndex>
	class SceneReconstructionEngine
	{
	public:
		/** Clear and reset a scene to set up a new empty
		    one.
		*/
		virtual void ResetScene(Scene<TVoxel, TIndex> *scene) = 0;

		/** Given a view with a new depth image, compute the
		    visible blocks, allocate them and update the hash
		    table so that the new image data can be integrated.
		*/
		virtual void AllocateSceneFromDepth(Scene<TVoxel, TIndex> *scene, const Intr intr, 
	const Affine3f pose, cuda::Dists &dist,const RenderState *renderState,bool onlyUpdateVisibleList = false,bool resetVisibleList = false)=0;

		/** Update the voxel blocks by integrating depth and
		    possibly colour information from the given view.
		*/
		virtual void IntegrateIntoScene(Scene<TVoxel, TIndex> *scene, const Intr intr,
	const Affine3f pose, cuda::Dists& dist, const RenderState *renderState)=0;
		SceneReconstructionEngine(void) { }
		virtual ~SceneReconstructionEngine(void) { }
	};
}
