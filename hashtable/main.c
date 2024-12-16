#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_hashtable.h"

#define NUM_ITEMS (1 << 24)

int main() {
    KeyValue* d_hashtable;

    // Create the hash table
    gpu_hashtable_create(&d_hashtable);

    // Prepare data for insertion
    KeyValue* h_kvs = (KeyValue*)malloc(sizeof(KeyValue) * NUM_ITEMS);
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        h_kvs[i].key = i;
        h_kvs[i].value = i * 2;
    }

    KeyValue* d_kvs;
    cudaMalloc((void**)&d_kvs, sizeof(KeyValue) * NUM_ITEMS);
    cudaMemcpy(d_kvs, h_kvs, sizeof(KeyValue) * NUM_ITEMS, cudaMemcpyHostToDevice);
    //printf("=================\n");

    // Insert data into the hash table
    gpu_hashtable_insert(d_hashtable, d_kvs, NUM_ITEMS);
    //printf("=================\n");

    // Prepare keys for lookup
    HASH_KEY_TYPE* h_keys = (HASH_KEY_TYPE*)malloc(sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        h_keys[i] = i;
    }

    HASH_KEY_TYPE* d_keys;
    HASH_VALUE_TYPE* d_values;
    HASH_VALUE_TYPE* h_values = (HASH_VALUE_TYPE*)malloc(sizeof(HASH_VALUE_TYPE) * NUM_ITEMS);

    cudaMalloc((void**)&d_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    cudaMalloc((void**)&d_values, sizeof(HASH_VALUE_TYPE) * NUM_ITEMS);
    cudaMemcpy(d_keys, h_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS, cudaMemcpyHostToDevice);

    //printf("=================\n");
    // Lookup keys in the hash table
    gpu_hashtable_lookup(d_hashtable, d_keys, d_values, NUM_ITEMS);

    // Copy results back to host
    cudaMemcpy(h_values, d_values, sizeof(HASH_VALUE_TYPE) * NUM_ITEMS, cudaMemcpyDeviceToHost);

    // Verify results
    uint32_t errors = 0;
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        if (h_values[i] != h_kvs[i].value) {
            ++errors;
        }
    }
    if (errors) {
        printf("Test failed with %u errors\n", errors);
    } else {
        printf("Test passed!\n");
    }

    // Clean up
    free(h_kvs);
    free(h_keys);
    free(h_values);
    cudaFree(d_kvs);
    cudaFree(d_keys);
    cudaFree(d_values);
    gpu_hashtable_destroy(d_hashtable);

    return 0;
}
