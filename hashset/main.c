#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_hashset.h"

#define NUM_ITEMS (1 << 24)

int main() {
    HASH_KEY_TYPE* d_hashset;

    // Create the hash set
    gpu_hashset_create(&d_hashset);

    // Prepare data for insertion
    HASH_KEY_TYPE* h_keys = (HASH_KEY_TYPE*)malloc(sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        h_keys[i] = i;
    }

    HASH_KEY_TYPE* d_keys;
    cudaMalloc((void**)&d_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    cudaMemcpy(d_keys, h_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS, cudaMemcpyHostToDevice);

    // Insert keys into the hash set
    gpu_hashset_insert(d_hashset, d_keys, NUM_ITEMS);

    // Prepare keys for contains check
    HASH_KEY_TYPE* h_check_keys = (HASH_KEY_TYPE*)malloc(sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        h_check_keys[i] = i;
    }

    HASH_KEY_TYPE* d_check_keys;
    uint8_t* d_results;
    uint8_t* h_results = (uint8_t*)malloc(sizeof(uint8_t) * NUM_ITEMS);

    cudaMalloc((void**)&d_check_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS);
    cudaMalloc((void**)&d_results, sizeof(uint8_t) * NUM_ITEMS);
    cudaMemcpy(d_check_keys, h_check_keys, sizeof(HASH_KEY_TYPE) * NUM_ITEMS, cudaMemcpyHostToDevice);

    // Check if keys are in the hash set
    gpu_hashset_contains(d_hashset, d_check_keys, d_results, NUM_ITEMS);

    // Copy results back to host
    cudaMemcpy(h_results, d_results, sizeof(uint8_t) * NUM_ITEMS, cudaMemcpyDeviceToHost);

    // Verify results
    uint32_t errors = 0;
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
        if (h_results[i] != 1) {
            ++errors;
        }
    }
    if (errors) {
        printf("Test failed with %u errors\n", errors);
    } else {
        printf("Test passed!\n");
    }

    // Clean up
    free(h_keys);
    free(h_check_keys);
    free(h_results);
    cudaFree(d_keys);
    cudaFree(d_check_keys);
    cudaFree(d_results);
    gpu_hashset_destroy(d_hashset);

    return 0;
}
