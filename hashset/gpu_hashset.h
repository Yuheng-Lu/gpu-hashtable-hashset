#ifndef GPU_HASHSET_H
#define GPU_HASHSET_H

#include <stdint.h>
#include <cuda.h>

// User-configurable definitions for key type
#ifndef HASH_KEY_TYPE
#define HASH_KEY_TYPE uint32_t
#endif

// Define empty key
#ifndef HASH_EMPTY_KEY
#define HASH_EMPTY_KEY 0xFFFFFFFF
#endif

// Hash set capacity (must be a power of two)
#ifndef HASH_SET_CAPACITY
#define HASH_SET_CAPACITY (1 << 27) // Example: 32 million entries
#endif

// Function prototypes
#ifdef __cplusplus
extern "C" {
#endif

void gpu_hashset_create(HASH_KEY_TYPE** d_hashset);

void gpu_hashset_destroy(HASH_KEY_TYPE* d_hashset);

void gpu_hashset_insert(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint32_t num_keys);

void gpu_hashset_delete(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint32_t num_keys);

void gpu_hashset_contains(HASH_KEY_TYPE* d_hashset, const HASH_KEY_TYPE* d_keys, uint8_t* d_results, uint32_t num_keys);

#ifdef __cplusplus
}
#endif

#endif // GPU_HASHSET_H
