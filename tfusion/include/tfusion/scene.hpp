#pragma once

#include "LocalVBA.hpp"
#include "GlobalCache.hpp"
#include "../../Utils/SceneParams.hpp"

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

		void SaveToDirectory(const std::string &outputDirectory) const
		{
			localVBA.SaveToDirectory(outputDirectory);
			index.SaveToDirectory(outputDirectory);
		}

		void LoadFromDirectory(const std::string &outputDirectory)
		{
			localVBA.LoadFromDirectory(outputDirectory);
			index.LoadFromDirectory(outputDirectory);			
		}

		Scene(const SceneParams *_sceneParams, bool _useSwapping, MemoryDeviceType _memoryType)
			: sceneParams(_sceneParams), index(_memoryType), localVBA(_memoryType, index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
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
