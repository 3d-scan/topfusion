#pragma once

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "Math.hpp"
#include "tfusion/types.hpp"

#define SDF_BLOCK_SIZE 8				// SDF block size
#define SDF_BLOCK_SIZE3 512				// SDF_BLOCK_SIZE3 = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE

// #define SDF_LOCAL_BLOCK_NUM 0x40000		// Number of locally stored blocks, currently 2^17
#define SDF_LOCAL_BLOCK_NUM 0x10000

#define SDF_BUCKET_NUM 0x100000			// Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM, SDF_HASH_MASK = SDF_BUCKET_NUM - 1
#define SDF_HASH_MASK 0xfffff			// Used for get hashing value of the bucket index,  SDF_HASH_MASK = SDF_BUCKET_NUM - 1
#define SDF_EXCESS_LIST_SIZE 0x20000	// 0x20000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

//// for loop closure
//#define SDF_LOCAL_BLOCK_NUM 0x10000		// Number of locally stored blocks, currently 2^12
//
//#define SDF_BUCKET_NUM 0x40000			// Number of Hash Bucket, should be 2^n and bigger than SDF_LOCAL_BLOCK_NUM, SDF_HASH_MASK = SDF_BUCKET_NUM - 1
//#define SDF_HASH_MASK 0x3ffff			// Used for get hashing value of the bucket index,  SDF_HASH_MASK = SDF_BUCKET_NUM - 1
//#define SDF_EXCESS_LIST_SIZE 0x8000		// 0x8000 Size of excess list, used to handle collisions. Also max offset (unsigned short) value.

#define SDF_TRANSFER_BLOCK_NUM 0x1000	// Maximum number of blocks transfered in one swap operation

/** \brief
	A single entry in the hash table.
*/
struct HashEntry
{
	/** Position of the corner of the 8x8x8 volume, that identifies the entry. */
	Vector3s pos;
	/** Offset in the excess list. */
	int offset;
	/** Pointer to the voxel block array.
		- >= 0 identifies an actual allocated entry in the voxel block array
		- -1 identifies an entry that has been removed (swapped out)
		- <-1 identifies an unallocated block
	*/
	int ptr;
};

namespace tfusion
{
	/** \brief
	This is the central class for the voxel block hash
	implementation. It contains all the data needed on the CPU
	and a pointer to the data structure on the GPU.
	*/
	class VoxelBlockHash
	{
	public:
		typedef HashEntry IndexData;

		struct IndexCache {
			Vector3i blockPos;
			int blockPtr;
			_CPU_AND_GPU_CODE_ IndexCache(void) : blockPos(0x7fffffff), blockPtr(-1) {}
		};

		/** Maximum number of total entries. */
		static const CONSTPTR(int) noTotalEntries = SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE;
		static const CONSTPTR(int) voxelBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	private:
		int lastFreeExcessListId;

		/** The actual data in the hash table. */
		// ORUtils::MemoryBlock<HashEntry> *hashEntries;
		cuda::DeviceArray<HashEntry> hashEntries;

		/** Identifies which entries of the overflow
		list are allocated. This is used if too
		many hash collisions caused the buckets to
		overflow.
		*/
		// ORUtils::MemoryBlock<int> *excessAllocationList;
		cuda::DeviceArray<int> excessAllocationList;


	public:
		VoxelBlockHash()
		{
			hashEntries.create(noTotalEntries * sizeof(HashEntry));
			excessAllocationList.create(SDF_EXCESS_LIST_SIZE * sizeof(int));
		}

		~VoxelBlockHash(void)
		{
			// delete hashEntries;
			// delete excessAllocationList;
		}

		/** Get the list of actual entries in the hash table. */
		const HashEntry *GetEntries(void) const { return hashEntries.ptr(); }
		HashEntry *GetEntries(void) { return hashEntries.ptr(); }

		const IndexData *getIndexData(void) const { return hashEntries.ptr(); }
		IndexData *getIndexData(void) { return hashEntries.ptr(); }

		/** Get the list that identifies which entries of the
		overflow list are allocated. This is used if too
		many hash collisions caused the buckets to overflow.
		*/
		const int *GetExcessAllocationList(void) const { return excessAllocationList.ptr(); }
		int *GetExcessAllocationList(void) { return excessAllocationList.ptr(); }

		int GetLastFreeExcessListId(void) { return lastFreeExcessListId; }
		void SetLastFreeExcessListId(int lastFreeExcessListId) { this->lastFreeExcessListId = lastFreeExcessListId; }

		/** Maximum number of total entries. */
		int getNumAllocatedVoxelBlocks(void) { return SDF_LOCAL_BLOCK_NUM; }
		int getVoxelBlockSize(void) { return SDF_BLOCK_SIZE3; }


		// Suppress the default copy constructor and assignment operator
		VoxelBlockHash(const VoxelBlockHash&);
		VoxelBlockHash& operator=(const VoxelBlockHash&);
	};
}
