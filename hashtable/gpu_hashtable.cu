#include "gpu_hashtable.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Hash function (e.g., MurmurHash3)
__device__ __host__ __forceinline__ uint32_t hash(HASH_KEY_TYPE key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

__global__ void gpu_hashtable_init_kernel(KeyValue* hashtable) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HASH_TABLE_CAPACITY) {
        hashtable[idx].key = HASH_EMPTY_KEY;
        hashtable[idx].value = HASH_EMPTY_VALUE;
    }
}

void gpu_hashtable_create(KeyValue** d_hashtable) {
    cudaError_t err = cudaMalloc((void**)d_hashtable, sizeof(KeyValue) * HASH_TABLE_CAPACITY);
    if (err != cudaSuccess) {
	        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
		    return;
    }

    uint32_t blockSize = 1024;
    uint32_t gridSize = (HASH_TABLE_CAPACITY + blockSize - 1) / blockSize;
    gpu_hashtable_init_kernel<<<gridSize, blockSize>>>(*d_hashtable);
    cudaDeviceSynchronize();
}

void gpu_hashtable_destroy(KeyValue* d_hashtable) {
    cudaFree(d_hashtable);
}

__device__ void gpu_hashtable_insert_device(KeyValue* hashtable, HASH_KEY_TYPE key, HASH_VALUE_TYPE value) {
    uint32_t slot = hash(key) & (HASH_TABLE_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE prev = atomicCAS(&hashtable[slot].key, HASH_EMPTY_KEY, key);
        if (prev == HASH_EMPTY_KEY || prev == key) {
            hashtable[slot].value = value;
            break;
        }
        slot = (slot + 1) & (HASH_TABLE_CAPACITY - 1);
    }
}

__global__ void gpu_hashtable_insert_kernel(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_kvs) {
        gpu_hashtable_insert_device(hashtable, kvs[idx].key, kvs[idx].value);
    }
}

void gpu_hashtable_insert(KeyValue* d_hashtable, const KeyValue* d_kvs, uint32_t num_kvs) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_kvs + blockSize - 1) / blockSize;
    gpu_hashtable_insert_kernel<<<gridSize, blockSize>>>(d_hashtable, d_kvs, num_kvs);
    cudaDeviceSynchronize();
}

__device__ HASH_VALUE_TYPE gpu_hashtable_lookup_device(KeyValue* hashtable, HASH_KEY_TYPE key) {
    uint32_t slot = hash(key) & (HASH_TABLE_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE curr = hashtable[slot].key;
        if (curr == key) {
            return hashtable[slot].value;
        }
        if (curr == HASH_EMPTY_KEY) {
            return HASH_EMPTY_VALUE;
        }
        slot = (slot + 1) & (HASH_TABLE_CAPACITY - 1);
    }
}

__global__ void gpu_hashtable_lookup_kernel(KeyValue* hashtable, const HASH_KEY_TYPE* keys, HASH_VALUE_TYPE* values, uint32_t num_keys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        values[idx] = gpu_hashtable_lookup_device(hashtable, keys[idx]);
    }
}

void gpu_hashtable_lookup(KeyValue* d_hashtable, const HASH_KEY_TYPE* d_keys, HASH_VALUE_TYPE* d_values, uint32_t num_keys) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_keys + blockSize - 1) / blockSize;
    gpu_hashtable_lookup_kernel<<<gridSize, blockSize>>>(d_hashtable, d_keys, d_values, num_keys);
    cudaDeviceSynchronize();
}

__device__ void gpu_hashtable_delete_device(KeyValue* hashtable, HASH_KEY_TYPE key) {
    uint32_t slot = hash(key) & (HASH_TABLE_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE curr = hashtable[slot].key;
        if (curr == key) {
            hashtable[slot].value = HASH_EMPTY_VALUE;
            break;
        }
        if (curr == HASH_EMPTY_KEY) {
            break;
        }
        slot = (slot + 1) & (HASH_TABLE_CAPACITY - 1);
    }
}

__global__ void gpu_hashtable_delete_kernel(KeyValue* hashtable, const HASH_KEY_TYPE* keys, uint32_t num_keys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        gpu_hashtable_delete_device(hashtable, keys[idx]);
    }
}

void gpu_hashtable_delete(KeyValue* d_hashtable, const HASH_KEY_TYPE* d_keys, uint32_t num_keys) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_keys + blockSize - 1) / blockSize;
    gpu_hashtable_delete_kernel<<<gridSize, blockSize>>>(d_hashtable, d_keys, num_keys);
    cudaDeviceSynchronize();
}
