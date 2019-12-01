#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#define MAX_MATRIX_WIDTH 4096

__constant__ int matrixWidth;
__constant__ int diagonal;

__global__ void gaussianEliminationKernel(float* matrix, float* vector)
{
    /*
        Sidenote for making reading the code easier:
        - Indexing the matrix is done using the formula: row*matrixWidth + column
    */
	int blockSize = blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadCount = blockDim.x * gridDim.x;

    // Divide the matrix rows between blocks. The division can't always be done evenly, so some blocks will receive more or less rows than others.
    // Amount of rows per forward or backwards substitution (for the entire kernel, not just this block)
    int rowsToReduce = matrixWidth - 1;

    // Amount of blocks that will receive one more row than others
    int burdenedBlocks = rowsToReduce % gridDim.x;

    // Amount of rows distributed to less burdened blocks
    int rowsPerBlock = rowsToReduce / gridDim.x;

    // Offset in rows from the current diagonal element's row
    int forwardRowOffset = rowsPerBlock * blockIdx.x;

    if (blockIdx.x < gridDim.x - burdenedBlocks) {
        // This is not a burdened block
        rowsToReduce = rowsPerBlock;
    } else {
        rowsToReduce = rowsPerBlock + 1;
        forwardRowOffset += blockIdx.x - (gridDim.x - burdenedBlocks);
    }

    int backRowOffset = forwardRowOffset + diagonal + 1 - matrixWidth;

    // Amount of rows to backwards substitute
    int backSubRows = max(0, min(rowsToReduce, rowsToReduce - (matrixWidth - (diagonal + 1 + forwardRowOffset))));

    // Amount of rows to forwards substitute
    int forwardSubRows = rowsToReduce - backSubRows;

    if (diagonal < matrixWidth) {
        // Division step
        float denominator = 1.0f / matrix[diagonal * matrixWidth + diagonal];

        // Calculate thread's index so that the first thread starts one row beneath the current diagonal element's row
        int blockStartingRow = diagonal + 1 + forwardRowOffset;
        // The block will handle all elements between blockFirstElement up until (but excluding) blockFinalElement
		int blockFinalElement = (blockStartingRow + forwardSubRows) * matrixWidth;
        int row = blockStartingRow + threadIdx.x;

        for (int idx = row * matrixWidth + diagonal; idx < blockFinalElement; idx += blockSize * matrixWidth) {
            float rowMultFactor = matrix[idx] * denominator;
            matrix[idx] = rowMultFactor;

            // Reduce vector
            vector[row] = vector[row] - rowMultFactor * vector[diagonal];

            row += blockSize;
        }

        // Perform the same divisions for all rows above the diagonal element
        blockStartingRow = diagonal - 1 - backRowOffset;
        blockFinalElement = (row - backSubRows) * matrixWidth + matrixWidth;
        row = blockStartingRow - threadIdx.x;

        for (int idx = row * matrixWidth + diagonal; idx > blockFinalElement; idx -= blockSize * matrixWidth) {
            float rowMultFactor = matrix[idx] * denominator;
            matrix[idx] = rowMultFactor;

            vector[row] = vector[row] - rowMultFactor * vector[diagonal];

            row -= blockSize;
        }

        // Every reduction factor for a block's rows needs to be stored before reduction can be performed
        __syncthreads();

        // Reduction step, row by row
        // Amount of elements to reduce per row (-1 as the elements beneath the diagonal elements are 'manually' set to 0)
        int elementsPerRow = matrixWidth - diagonal - 1;

        // Pre-calculations for forward and backwards reduction
        int threadRowOffset = tid / elementsPerRow;
        int threadColumnOffset = tid - threadRowOffset * elementsPerRow;

        // The base value for iterating through the matrix elements is to start at [row, column] = [diagonal+1, diagonal+1]
        // Calculate an offset from this position based on the thread ID
        if (forwardSubRows > 0) {
            int blockStartingElement = (diagonal + 1 + forwardRowOffset) * matrixWidth + diagonal + 1;
			blockFinalElement = blockStartingElement + matrixWidth * forwardSubRows;

			int idx = blockStartingElement + threadRowOffset * matrixWidth + threadColumnOffset;
            int elementNr = tid;

            // Forward substitution (reducing rows downwards)
            while (idx < blockFinalElement) {
				// Current element - element in the diagonal element's row, above idx * row's multiplying factor stored in the current row (underneath the diagonal element)
                matrix[idx] -= matrix[matrixWidth * diagonal + (diagonal + threadColumnOffset + 1)] * matrix[idx - threadColumnOffset - 1];

                // Transform the index to dodge already reduced rows and columns to the left of the diagonal elmement
                elementNr += threadCount;

                threadRowOffset = elementNr / elementsPerRow;
				threadColumnOffset = elementNr - threadRowOffset * elementsPerRow;

				idx = blockStartingElement + threadRowOffset * matrixWidth + threadColumnOffset;
            }
        }

        if (backSubRows > 0) {
            // Backwards substitution (upwards reducing)
			int blockStartingElement = backRowOffset * matrixWidth + diagonal + 1;
			blockFinalElement = blockStartingElement + matrixWidth * backSubRows;

			int idx = blockStartingElement + threadRowOffset * matrixWidth + threadColumnOffset;
            int elementNr = tid;

            while (idx < blockFinalElement) {
				// Current element - element in the diagonal element's row, above idx * row's multiplying factor stored in the current row (underneath the diagonal element)
                matrix[idx] -= matrix[matrixWidth * diagonal + (diagonal + threadColumnOffset + 1)] * matrix[idx - threadColumnOffset - 1];

                // Transform the index to dodge already reduced rows and columns to the left of the diagonal elmement
                elementNr += threadCount;

                threadRowOffset = elementNr / elementsPerRow;
				threadColumnOffset = elementNr - threadRowOffset * elementsPerRow;

				idx = blockStartingElement + threadRowOffset * matrixWidth + threadColumnOffset;
            }
        }

    } else {
        // Normalize diagonal
		tid = blockIdx.x * blockDim.x + threadIdx.x;
		int threadCount = blockDim.x * gridDim.x;

        for (int diagonalElement = tid; diagonalElement < matrixWidth; diagonalElement += threadCount) {
            int diagIdx = diagonalElement * matrixWidth + diagonalElement;
            float diagValue = matrix[diagIdx];

            vector[diagonalElement] /= diagValue;
        }
    }
}

enum Program {
    CPU = 0,
    CUDA = 1
};

struct Configuration {
    Program program;
    unsigned int threadCount;
    unsigned int matrixWidth;
    bool quick;
};

void initConfig(Configuration& config, int argCount, char* argValues[]);
void initMatrix(std::vector<float>& matrix, std::vector<float>& vector, unsigned int matrixWidth);
void cpuGaussianEliminatation(std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& solutionVec, unsigned int matrixWidth);
cudaError_t cudaGaussianElimination(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& solutionVec, const Configuration& config);
void checkValidity(const std::vector<float>& matrix, const std::vector<float>& solutionVec, const std::vector<float>& rightHandVec, unsigned int matrixWidth);
void printMatrix(const std::vector<float>& matrix, const std::vector<float>& vector, const std::string& fileName, unsigned int matrixWidth);

int main(int argCount, char* argValues[])
{
    Configuration config;
    initConfig(config, argCount, argValues);

    std::vector<float> matrix, vector, solutionVec;
    initMatrix(matrix, vector, config.matrixWidth);
    if (!config.quick) {
        printMatrix(matrix, vector, "matrixIn", config.matrixWidth);
    }

    printf("Starting gaussian elimination\n");
    if (config.program == CPU) {
        cpuGaussianEliminatation(matrix, vector, solutionVec, config.matrixWidth);
    } else {
        cudaError_t cudaStatus = cudaGaussianElimination(matrix, vector, solutionVec, config);
        if (cudaStatus != cudaSuccess) {
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
    }

    printf("Finished gaussian elimination\n");

    if (!config.quick) {
        checkValidity(matrix, solutionVec, vector, config.matrixWidth);
        printMatrix(matrix, solutionVec, "matrixOut", config.matrixWidth);
    }

    return 0;
}

void initConfig(Configuration& config, int argCount, char* argValues[])
{
    // Initialize default values
    config.program = CUDA;
    config.threadCount = 8192;
    config.matrixWidth = 200;
    config.quick = false;

    for (int argIdx = 1; argIdx < argCount; argIdx += 1) {
        char* arg = argValues[argIdx];

        if (arg[0] != '-') {
            continue;
        }

        char flag = arg[1];
        std::string argStr = "";

        switch(flag) {
            case 'p':
                argStr = std::string(argValues[++argIdx]);
                if (argStr == "CPU") {
                    config.program = CPU;
                } else if (argStr == "CUDA") {
                    config.program = CUDA;
                }
                break;

            case 't':
                argStr = std::string(argValues[++argIdx]);
                config.threadCount = std::stoi(argStr);
                break;

            case 'n':
                argStr = std::string(argValues[++argIdx]);
                config.matrixWidth = std::stoi(argStr);
                break;

            // q for 'quick'
            case 'q':
                config.quick = true;
                break;

            default:
                printf("Failed to parse flag: '%c'\n", flag);
                break;
        }
    }
}

void initMatrix(std::vector<float>& matrix, std::vector<float>& vector, unsigned int matrixWidth)
{
    srand((unsigned int)time(NULL));
    unsigned int matrixSize = matrixWidth * matrixWidth;
    matrix.resize(matrixSize);
    vector.resize(matrixWidth);

    for (size_t row = 0; row < matrixWidth; row += 1) {
        size_t rowStartIdx = row * matrixWidth;

        for (size_t col = 0; col < matrixWidth; col += 1) {
            matrix[rowStartIdx + col] = (float)(rand() % 5) + 1.0f;

            // Make sure diagonal elements are larger than non-diagonal elements
            if (row == col) {
                matrix[rowStartIdx + col] += 5;
            }
        }
    }

    for (unsigned int i = 0; i < matrixWidth; i += 1) {
        vector[i] = (float)(rand() % 3) + 1.0f;
    }
}

void cpuGaussianEliminatation(std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& solutionVec, unsigned int matrixWidth)
{
    solutionVec.resize(vector.size());
    std::memcpy(&solutionVec.front(), &vector.front(), sizeof(float) * vector.size());

    std::vector<float> matrixCpy;
    matrixCpy.resize(matrix.size());
    std::memcpy(&matrixCpy.front(), &matrix.front(), sizeof(float) * matrix.size());

    for (size_t diagonal = 0; diagonal < matrixWidth; diagonal += 1) {
        float diagonalReciprocal = 1.0f / matrixCpy[diagonal * matrixWidth + diagonal];

        // Forward substitution
        for (size_t subRow = diagonal + 1; subRow < matrixWidth; subRow += 1) {
            size_t rowStartIdx = subRow * matrixWidth;
            float multFactor = matrixCpy[rowStartIdx + diagonal] * diagonalReciprocal;

            for (size_t subColumn = diagonal + 1; subColumn < matrixWidth; subColumn += 1) {
                matrixCpy[rowStartIdx + subColumn] -= multFactor * matrixCpy[diagonal * matrixWidth + subColumn];
            }

            // Reduce right-hand vector
            solutionVec[subRow] -= multFactor * solutionVec[diagonal];
        }

        // Backward substitution
        for (int subRow = (int)diagonal - 1; subRow >= 0; subRow -= 1) {
            size_t rowStartIdx = (size_t)subRow * matrixWidth;
            float multFactor = matrixCpy[rowStartIdx + diagonal] * diagonalReciprocal;

            for (size_t subColumn = diagonal+ 1; subColumn < matrixWidth; subColumn += 1) {
                matrixCpy[rowStartIdx + subColumn] -= multFactor * matrixCpy[diagonal * matrixWidth + subColumn];
            }

            // Reduce right-hand vector
            solutionVec[subRow] -= multFactor * solutionVec[diagonal];
        }
    }

    // Normalize diagonal
    for (size_t diagonal = 0; diagonal < matrixWidth; diagonal += 1) {
        solutionVec[diagonal] /= matrixCpy[diagonal * matrixWidth + diagonal];
    }

    //matrix = matrixCpy;
}

cudaError_t cudaGaussianElimination(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& solutionVec, const Configuration& config)
{
    float* deviceMatrix = nullptr, *deviceVector = nullptr;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate GPU buffer for matrix.
    cudaStatus = cudaMalloc((void**)&deviceMatrix, matrix.size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate GPU buffer for vector.
    cudaStatus = cudaMalloc((void**)&deviceVector, vector.size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Send matrix width variable to GPU.
    int mWidth = (int)config.matrixWidth;
    cudaStatus = cudaMemcpyToSymbol(matrixWidth, (void*)&mWidth, sizeof(mWidth));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy matrix to GPU buffer.
    cudaStatus = cudaMemcpy(deviceMatrix, &matrix.front(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // Copy vector to GPU buffer.
    cudaStatus = cudaMemcpy(deviceVector, &vector.front(), vector.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // Get CUDA device properties
    cudaDeviceProp deviceProperties;

    cudaStatus = cudaGetDeviceProperties(&deviceProperties, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed to get CUDA device properties: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    printf("Shared memory per block: %zd\n", deviceProperties.sharedMemPerBlock);
    printf("Max threads per block: %d\n", deviceProperties.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", deviceProperties.maxThreadsPerMultiProcessor);
    printf("Multiprocessors: %d\n", deviceProperties.multiProcessorCount);
    printf("Registries per block: %d\n", deviceProperties.regsPerBlock);

    // Limit thread count if it exceeds the GPU's capacity
    unsigned int maxThreads = deviceProperties.maxThreadsPerMultiProcessor * deviceProperties.multiProcessorCount;
    unsigned int threadCount = std::min(maxThreads, config.threadCount);

    // Use the requested thread count to calculate the amount of blocks needed, and the amount of threads per block
    unsigned int blockCount = 1 + (threadCount-1) / (unsigned int)deviceProperties.maxThreadsPerBlock;

    /*
        The requested amount of thread might not be evenly distributed into the blocks, eg. 7 threads can't be
        evenly divided among 3 blocks. Calculate the amount of threads to add to avoid this issue.
    */
    unsigned int threadsToAdd = threadCount % blockCount;
    unsigned int threadsPerBlock = (threadCount + threadsToAdd) / blockCount;

    printf("Blocks: %d, Threads: %d (added %d to the requested amount)\n", blockCount, threadCount + threadsToAdd, threadsToAdd);

    // Launch the kernel on the GPU.
    const dim3 gridDim = {blockCount, 1, 1};
    const dim3 blockDim = {threadsPerBlock, 1, 1};

    for (int hDiagonal = 0; hDiagonal <= (int)config.matrixWidth; hDiagonal += 1) {
        // Send matrix width variable to GPU.
        cudaStatus = cudaMemcpyToSymbol(diagonal, (void*)&hDiagonal, sizeof(hDiagonal));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        gaussianEliminationKernel<<<gridDim, blockDim, 0>>>(deviceMatrix, deviceVector);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            goto Error;
        }
    }

    // Copy output vector from GPU buffer to host memory.
    solutionVec.resize(vector.size());

    cudaStatus = cudaMemcpy(&solutionVec.front(), deviceVector, solutionVec.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(deviceMatrix);

    return cudaStatus;
}

void checkValidity(const std::vector<float>& matrix, const std::vector<float>& solutionVec, const std::vector<float>& rightHandVec, unsigned int matrixWidth)
{
    for (unsigned int row = 0; row < matrixWidth; row += 1) {
        float rowSum = 0.0f;

        unsigned int rowIdx = row * matrixWidth;

        for (unsigned int col = 0; col < matrixWidth; col += 1) {
            rowSum += matrix[rowIdx + col] * solutionVec[col];
        }

        if (std::abs(rowSum - rightHandVec[row]) > 0.1f) {
            printf("Solution is wrong! Row %d expected value: %f, actual: %f\n", row, rightHandVec[row], rowSum);
            return;
        }
    }

    printf("Solution is correct!\n");
}

void printMatrix(const std::vector<float>& matrix, const std::vector<float>& vector, const std::string& fileName, unsigned int matrixWidth)
{
    // Output matrix to file
    std::ofstream fileOut;
    fileOut.open(fileName, std::ios::out);

    for (unsigned int row = 0; row < matrixWidth; row += 1) {
        for (unsigned int col = 0; col < matrixWidth; col += 1) {
            fileOut << std::left << std::setfill(' ') << std::setw(11) << matrix[row * matrixWidth + col];
        }

        fileOut << '\n';
    }

    fileOut << '\n';

    for (float vecResult : vector) {
        fileOut << vecResult << "  ";
    }

    if (matrixWidth < 7) {
        // Print the equation in a Wolfram-Alpha-friendly format for small matrices
        fileOut << "\n\n{";

        // Print matrix
        for (unsigned int row = 0; row < matrixWidth; row += 1) {
            fileOut << '{';
            fileOut << matrix[row * matrixWidth];

            for (unsigned int col = 1; col < matrixWidth; col += 1) {
                fileOut << "," << matrix[row * matrixWidth + col];
            }

            fileOut << '}';

            if (row + 1 < matrixWidth) {
                fileOut << ',';
            }
        }

        fileOut << "}";

        // Print variable vector
        fileOut << "*{a";

        for (unsigned int varIdx = 1; varIdx < matrixWidth; varIdx += 1) {
            fileOut << ',' << (char)('a' + varIdx);
        }

        fileOut << "}";

        // Print right-hand vector
        fileOut << "={" << vector.front();

        for (unsigned int vecIdx = 1; vecIdx < matrixWidth; vecIdx += 1) {
            fileOut << ',' << vector[vecIdx];
        }

        fileOut << '}';
    }

    fileOut.close();
}
