#pragma once

#include "tfusion/LocalVBA.hpp"
#include "tfusion/GlobalCache.hpp"
#include "tfusion/SceneParams.hpp"

namespace tfusion
{
	/** \brief
	Represents the 3D world model as a hash of small voxel
	blocks
	*/
	template<class TVoxel, class TIndex>
	class Scene
	{
	public:
		/** Scene parameters like voxel size etc. */
		const SceneParams *sceneParams;

		/** Hash table to reference the 8x8x8 blocks */
		TIndex index;

		/** Current local content of the 8x8x8 voxel blocks -- stored host or device */
		LocalVBA<TVoxel> localVBA;

		/** Global content of the 8x8x8 voxel blocks -- stored on host only */
		GlobalCache<TVoxel> *globalCache;

		Scene(const SceneParams *_sceneParams, bool _useSwapping)
			: sceneParams(_sceneParams), localVBA(index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
		{
			if (_useSwapping) globalCache = new GlobalCache<TVoxel>();
			else globalCache = NULL;
		}

		~Scene(void)
		{
			if (globalCache != NULL) delete globalCache;
		}

		// Suppress the default copy constructor and assignment operator
		Scene(const Scene&);
		Scene& operator=(const Scene&);
	};
}
