#include "gpu_hashset.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Hash function
__device__ __host__ __forceinline__ uint32_t hash(HASH_KEY_TYPE key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Kernel to initialize the hash set
__global__ void gpu_hashset_init_kernel(HASH_KEY_TYPE* hashset) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HASH_SET_CAPACITY) {
        hashset[idx] = HASH_EMPTY_KEY;
    }
}

void gpu_hashset_create(HASH_KEY_TYPE** d_hashset) {
    cudaMalloc((void**)d_hashset, sizeof(HASH_KEY_TYPE) * HASH_SET_CAPACITY);

    uint32_t blockSize = 1024;
    uint32_t gridSize = (HASH_SET_CAPACITY + blockSize - 1) / blockSize;
    gpu_hashset_init_kernel<<<gridSize, blockSize>>>(*d_hashset);
    cudaDeviceSynchronize();
}

void gpu_hashset_destroy(HASH_KEY_TYPE* d_hashset) {
    cudaFree(d_hashset);
}

__device__ void gpu_hashset_insert_device(HASH_KEY_TYPE* hashset, HASH_KEY_TYPE key) {
    uint32_t slot = hash(key) & (HASH_SET_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE prev = atomicCAS(&hashset[slot], HASH_EMPTY_KEY, key);
        if (prev == HASH_EMPTY_KEY || prev == key) {
            // Key inserted or already exists
            break;
        }
        slot = (slot + 1) & (HASH_SET_CAPACITY - 1);
    }
}

__global__ void gpu_hashset_insert_kernel(HASH_KEY_TYPE* hashset, const HASH_KEY_TYPE* keys, uint32_t num_keys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        gpu_hashset_insert_device(hashset, keys[idx]);
    }
}

void gpu_hashset_insert(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint32_t num_keys) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_keys + blockSize - 1) / blockSize;
    gpu_hashset_insert_kernel<<<gridSize, blockSize>>>(d_hashset, d_keys, num_keys);
    cudaDeviceSynchronize();
}

__device__ void gpu_hashset_delete_device(HASH_KEY_TYPE* hashset, HASH_KEY_TYPE key) {
    uint32_t slot = hash(key) & (HASH_SET_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE curr = hashset[slot];
        if (curr == key) {
            // Mark the key as deleted
            hashset[slot] = HASH_EMPTY_KEY;
            break;
        }
        if (curr == HASH_EMPTY_KEY) {
            // Key not found
            break;
        }
        slot = (slot + 1) & (HASH_SET_CAPACITY - 1);
    }
}

__global__ void gpu_hashset_delete_kernel(HASH_KEY_TYPE* hashset, const HASH_KEY_TYPE* keys, uint32_t num_keys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        gpu_hashset_delete_device(hashset, keys[idx]);
    }
}

void gpu_hashset_delete(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint32_t num_keys) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_keys + blockSize - 1) / blockSize;
    gpu_hashset_delete_kernel<<<gridSize, blockSize>>>(d_hashset, d_keys, num_keys);
    cudaDeviceSynchronize();
}

__device__ uint8_t gpu_hashset_contains_device(HASH_KEY_TYPE* hashset, HASH_KEY_TYPE key) {
    uint32_t slot = hash(key) & (HASH_SET_CAPACITY - 1);

    while (true) {
        HASH_KEY_TYPE curr = hashset[slot];
        if (curr == key) {
            // Key found
            return 1;
        }
        if (curr == HASH_EMPTY_KEY) {
            // Key not found
            return 0;
        }
        slot = (slot + 1) & (HASH_SET_CAPACITY - 1);
    }
}

__global__ void gpu_hashset_contains_kernel(HASH_KEY_TYPE* hashset, const HASH_KEY_TYPE* keys, uint8_t* results, uint32_t num_keys) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        results[idx] = gpu_hashset_contains_device(hashset, keys[idx]);
    }
}

void gpu_hashset_contains(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint8_t* d_results, uint32_t num_keys) {
    uint32_t blockSize = 1024;
    uint32_t gridSize = (num_keys + blockSize - 1) / blockSize;
    gpu_hashset_contains_kernel<<<gridSize, blockSize>>>(d_hashset, d_keys, d_results, num_keys);
    cudaDeviceSynchronize();
}
