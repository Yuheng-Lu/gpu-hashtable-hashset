# CUDA Hash Sets and Hash Tables

This project contains implementations of hash sets and hash tables for GPU environments.

## Directory Structure

```
.
├── hashset
│ ├── Makefile - Build instructions for GPU hash set.
│ ├── gpu_hashset.cu - Implementation of the GPU hash set (CUDA).
│ ├── gpu_hashset.h - Header file for the GPU hash set.
│ └── main.c - Main program for demonstrating the GPU hash set.
└── hashtable
  ├── Makefile - Build instructions for GPU hash table.
  ├── gpu_hashtable.cu - Implementation of the GPU hash table (CUDA).
  ├── gpu_hashtable.h - Header file for the GPU hash table.
  └── main.c - Main program for demonstrating the GPU hash table.
```

## Instructions

1. Building the project:

   - Navigate to the desired directory (e.g., hashset) and run `make` to build the test executable.

3. Running the main program:

   - Use the following commands:
   
     ```bash
     nvcc -c gpu_hashset.cu -o gpu_hashset.o
     gcc -c main.c -o main.o
     nvcc main.o gpu_hashset.o -o main
     ```
   
   - After compiling, execute the resulting main binary (e.g., ./main).
