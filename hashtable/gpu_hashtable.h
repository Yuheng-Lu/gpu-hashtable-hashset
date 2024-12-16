#ifndef GPU_HASHTABLE_H
#define GPU_HASHTABLE_H

#include <stdint.h>
#include <cuda.h>

// User-configurable definitions for key and value types
#ifndef HASH_KEY_TYPE
#define HASH_KEY_TYPE uint32_t
#endif

#ifndef HASH_VALUE_TYPE
#define HASH_VALUE_TYPE char
#endif

// Define empty key and value
#ifndef HASH_EMPTY_KEY
#define HASH_EMPTY_KEY NULL
#endif

#ifndef HASH_EMPTY_VALUE
#define HASH_EMPTY_VALUE NULL
#endif

// Hash table capacity (must be a power of two)
#ifndef HASH_TABLE_CAPACITY
#define HASH_TABLE_CAPACITY (1 << 27) // Example: 32 million entries
#endif

// KeyValue pair structure
typedef struct {
    HASH_KEY_TYPE key;
    HASH_VALUE_TYPE value;
} KeyValue;

// Function prototypes
#ifdef __cplusplus
extern "C" {
#endif

void gpu_hashtable_create(KeyValue** d_hashtable);

void gpu_hashtable_destroy(KeyValue* d_hashtable);

void gpu_hashtable_insert(KeyValue* d_hashtable, const KeyValue* d_kvs, uint32_t num_kvs);

void gpu_hashtable_delete(KeyValue* d_hashtable, const HASH_KEY_TYPE* d_keys, uint32_t num_keys);

void gpu_hashtable_lookup(KeyValue* d_hashtable, const HASH_KEY_TYPE* d_keys, HASH_VALUE_TYPE* d_values, uint32_t num_keys);

#ifdef __cplusplus
}
#endif

#endif // GPU_HASHTABLE_H
