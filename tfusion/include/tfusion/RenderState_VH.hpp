#pragma once

#include <stdlib.h>

#include "RenderState.h"
#include "../Scene/VoxelBlockHash.h"
#include "../../../ORUtils/MemoryBlock.h"

namespace tfusion
{
	/** \brief
	    Stores the render state used by the SceneReconstruction 
	    and visualisation engines, as used by voxel hashing.
	*/
	class RenderState_VH : public RenderState
	{
	private:
		// MemoryDeviceType memoryType;

		/** A list of "visible entries", that are currently
		being processed by the tracker.
		*/
		cuda::cudaData_array<int> visibleEntryIDs;

		/** A list of "visible entries", that are
		currently being processed by integration
		and tracker.
		*/
		cuda::cudaData_array<uchar> entriesVisibleType;
           
	public:
		/** Number of entries in the live list. */
		int noVisibleEntries;
           
		RenderState_VH(int noTotalEntries, const Vector2i & imgSize, float vf_min, float vf_max, MemoryDeviceType memoryType = MEMORYDEVICE_CPU)
			: RenderState(imgSize, vf_min, vf_max)
		{
			// this->memoryType = memoryType;

			// visibleEntryIDs = new ORUtils::MemoryBlock<int>(SDF_LOCAL_BLOCK_NUM, memoryType);
			// entriesVisibleType = new ORUtils::MemoryBlock<uchar>(noTotalEntries, memoryType);
			GetVisibleEntryIDs.create(SDF_LOCAL_BLOCK_NUM * sizeof(int));
			entriesVisibleType.create(noTotalEntries * sizeof(uchar));

			noVisibleEntries = 0;
		}
		~RenderState_VH()
		{
			delete visibleEntryIDs;
			delete entriesVisibleType;
		}
		/** Get the list of "visible entries", that are currently
		processed by the tracker.
		*/
		const int *GetVisibleEntryIDs(void) const { return visibleEntryIDs.ptr();}
		int *GetVisibleEntryIDs(void) { return visibleEntryIDs.ptr(); }
		/** Get the list of "visible entries", that are
		currently processed by integration and tracker.
		*/
		uchar *GetEntriesVisibleType(void) { return entriesVisibleType.ptr(); }
	};
} 
