#pragma once
#include <tfusion/types.hpp>

namespace tfusion
{
	/** \brief
	Stores the actual voxel content that is referred to by a
	Lib::HashTable.
	*/
	template<class TVoxel>
	class LocalVBA
	{
	private:
		// ORUtils::MemoryBlock<TVoxel> *voxelBlocks;
		// ORUtils::MemoryBlock<int> *allocationList;

		// MemoryDeviceType memoryType;
		tfusion::cuda::DeviceArray<TVoxel> voxelBlocks;
		tfusion::cuda::DeviceArray<int> allocationList;

	public:
		inline TVoxel *GetVoxelBlocks(void) {return voxelBlocks.ptr();}
		inline const TVoxel *GetVoxelBlocks(void) const { return voxelBlocks.ptr(); }
		int *GetAllocationList(void) { return allocationList.ptr(); }

		int lastFreeBlockId;

		int allocatedSize;

		LocalVBA(int noBlocks, int blockSize)
		{
			allocatedSize = noBlocks * blockSize;

			voxelBlocks.create(allocatedSize * sizeof(TVoxel));
			allocationList.create(noBlocks * sizeof(int));
		}
		~LocalVBA(void)
		{
			// delete voxelBlocks;
			// delete allocationList;
		}

		// Suppress the default copy constructor and assignment operator
		LocalVBA(const LocalVBA&);
		LocalVBA& operator=(const LocalVBA&);
	};
}