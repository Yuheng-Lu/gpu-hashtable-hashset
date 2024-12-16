# Compiler settings
CC = gcc                 # Compiler for C files
NVCC = nvcc              # Compiler for CUDA files

# Target executable name
TARGET = test            # Name of the final executable

# Source files and headers
C_SOURCES = test.c
CU_SOURCES = gpu_hashset.cu
HEADERS = gpu_hashset.h

# Object files (intermediate compiled files)
C_OBJECTS = $(C_SOURCES:.c=.o)      # Convert .c sources to .o objects
CU_OBJECTS = $(CU_SOURCES:.cu=.o)   # Convert .cu sources to .o objects

# Compiler flags
CFLAGS = -I. -g          # Include current directory for headers, enable debugging
NVCCFLAGS = -I. -g       # Same as above for CUDA files

# Linker flags (add CUDA libraries if needed)
LDFLAGS = -lcudart       # Link CUDA runtime library

# Default target: Build the executable
all: $(TARGET)

# Rule to link object files and create the final executable
$(TARGET): $(C_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

# Rule to compile C source files into object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA source files into object files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(C_OBJECTS) $(CU_OBJECTS) $(TARGET)
