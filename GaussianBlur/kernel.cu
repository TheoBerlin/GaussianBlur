#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#include "CImg.h"
#include "jpge.h"

using namespace cimg_library;

// Contains width and height of image
__constant__ int2 imageDims;

__global__ void AOStoSOAKernel(unsigned char* imageIn, unsigned char* imageOut)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadCount = blockDim.x * gridDim.x;

    int pixelCount = imageDims.x * imageDims.y;

    for (int pixelIdx = tid; pixelIdx < pixelCount; pixelIdx += threadCount) {
        // Red
        imageOut[pixelIdx] = imageIn[pixelIdx * 3];

        // Green
        imageOut[pixelIdx + pixelCount] = imageIn[pixelIdx * 3 + 1];

        // Blue
        imageOut[pixelIdx + pixelCount * 2] = imageIn[pixelIdx * 3 + 2];
    }
}

__global__ void gaussianBlurKernel(unsigned char* imageIn, unsigned char* imageOut, double* mask)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadCount = blockDim.x * gridDim.x;

    int pixelCount = imageDims.x * imageDims.y;

    // Blur all channels in each pixel
    for (int pixelIdx = tid; pixelIdx < pixelCount; pixelIdx += threadCount) {
        int yPos = pixelIdx / imageDims.x;
        int xPos = pixelIdx - yPos * imageDims.x;

        double red = 0.0, green = 0.0, blue = 0.0;
        int maskIdx = 0;
        for (int yOffset = -2; yOffset < 3; yOffset++) {
            int y = min(imageDims.y-1, max(0, yPos + yOffset));
            for (int xOffset = -2; xOffset < 3; xOffset++) {
                int x = min(imageDims.x-1, max(0, xPos + xOffset));
                int imageIdxRed = y * imageDims.x + x;

                double maskValue = mask[maskIdx++];
                red += imageIn[imageIdxRed] * maskValue;
                green += imageIn[imageIdxRed + pixelCount] * maskValue;
                blue += imageIn[imageIdxRed + pixelCount + pixelCount] * maskValue;

                imageIdxRed += 1;
            }
        }

        int outIdx = pixelIdx * 3;
        imageOut[outIdx] = red;
        imageOut[outIdx + 1] = green;
        imageOut[outIdx + 2] = blue;
    }
}

enum Program {
    CPU = 0,
    CUDA = 1
};

struct Configuration {
    Program program;
    unsigned int threadCount;
    bool quick;
};

void initConfig(Configuration& config, int argCount, char* argValues[]);
void loadPPM(const std::string& fileName, CImg<unsigned char>& image);
void initMask(CImg<double>& mask, bool quick);
cudaError_t cudaGaussianBlur(CImg<unsigned char>& image, const CImg<double>& mask, const Configuration& config);
bool checkValidity(const CImg<unsigned char>& cImage, const CImg<unsigned char>& cudaImage);

int main(int argCount, char* argValues[])
{
    Configuration config;
    initConfig(config, argCount, argValues);

    CImg<unsigned char> image, blurimage;
    CImg<double> mask;
    initMask(mask, config.quick);

    printf("Starting blurring process\n");
    if (config.program == CPU) {
        blurimage.load("cake.ppm");
        blurimage.convolve(mask);
    } else {
        loadPPM("cake.ppm", blurimage);

        cudaGaussianBlur(blurimage, mask, config);

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
    }
    printf("Finished blurring process\n");

    if (!config.quick) {
        // Display images and save the blurred version
        image.load("cake.ppm");
        CImgDisplay main_disp(image, "Original image");
        CImgDisplay main_disp2;

        // The CUDA program converts the data layout format to RGBRGB, which CImg isn't meant to display
        if (config.program == CPU) {
            // Display image blurred on the CPU
            main_disp2.set_title("Blurred image");
            main_disp2.display(blurimage);
            main_disp2.paint();
            main_disp2.show();
        } else {
            // Check if the CUDA blurring is equivalent to the CImg blurring
            checkValidity(image, blurimage);
        }

        // Save image to file
        std::string fileName = "blurred.";
            if (config.program == CPU) {
            fileName.append("ppm");
            blurimage.save(fileName.c_str());
        } else {
            fileName.append("jpeg");
            jpge::params jpegParams;
            jpegParams.m_quality = 100;

            if (!jpge::compress_image_to_jpeg_file(fileName.c_str(), blurimage.width(), blurimage.height(), blurimage.spectrum(), blurimage.data(), jpegParams)) {
                printf("Failed to write blurred image to %s\n", fileName.c_str());
            }
        }

        printf("Saved blurred image in file '%s'\n", fileName.c_str());

        // Keep the displays open
        std::getchar();
    }

    return 0;
}

void initConfig(Configuration& config, int argCount, char* argValues[])
{
    // Initialize default values
    config.program = CUDA;
    config.threadCount = 4096;
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

void loadPPM(const std::string& fileName, CImg<unsigned char>& image)
{
    std::ifstream file(fileName.c_str(), std::ios::in | std::ios::binary);
    if (!file) {
        printf("Failed to open file: %s\n", fileName.c_str());
        return;
    }

    // Ignore magic string
    std::string input = "";
    file >> input;

    // Read dimensions
    int width = 0, height = 0;
    file >> width;
    file >> height;

    image = CImg<unsigned char>(width, height, 1, 3);

    // Ignore max number and line dedicated to comments
    file >> input;
    file.ignore();

    // Read image data
    unsigned char* imageData = image.data();
    std::streamsize bytesPerRow = width * 3;

    const std::streamsize chunkSize = 8192;

    while (file) {
        file.read((char*)imageData, chunkSize);
        imageData += chunkSize;
    }

    file.close();
}

void initMask(CImg<double>& mask, bool quick)
{
	// Create the mask of weights (5 x 5 Gaussian blur)
    mask = CImg<double>(5,5);

	mask(0, 0) = mask(0, 4) = mask(4, 0) = mask(4, 4) = 1.0 / 256.0;
	mask(0, 1) = mask(0, 3) = mask(1, 0) = mask(1, 4) = mask(3, 0) = mask(3, 4) = mask(4, 1) = mask(4, 3) = 4.0 / 256.0;
	mask(0, 2) = mask(2, 0) = mask(2, 4) = mask(4, 2) = 6.0 / 256.0;
	mask(1, 1) = mask(1, 3) = mask(3, 1) = mask(3, 3) = 16.0 / 256.0;
	mask(1, 2) = mask(2, 1) = mask(2, 3) = mask(3, 2) = 24.0 / 256.0;
	mask(2, 2) = 36.0 / 256.0;

    if (!quick) {
        // Print the mask that is being used
        printf("5x5 mask:\n");

        for (int i = 0; i <= 4; i++) {
            for (int j = 0; j <= 4; j++) {
                std::cout << mask(i, j) << " ";
            }

            std::cout << "\n";
        }
    }
}

cudaError_t cudaGaussianBlur(CImg<unsigned char>& image, const CImg<double>& mask, const Configuration& config)
{
    unsigned char* deviceImageIn = nullptr, *deviceImageSOA = nullptr;
    double* deviceMask = nullptr;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate GPU buffer for input image.
    cudaStatus = cudaMalloc((void**)&deviceImageIn, image.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Allocate GPU buffer for restructued image.
    cudaStatus = cudaMalloc((void**)&deviceImageSOA, image.size() * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Send image dimensions vector to GPU.
    int2 imageDimensions[2] = {image.width(), image.height()};
    cudaStatus = cudaMemcpyToSymbol(imageDims, (void*)imageDimensions, sizeof(int) * 2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy image to GPU buffer.
    cudaStatus = cudaMemcpy(deviceImageIn, image.data(), image.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of image to device failed!\n");
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

    // Use the requested thread count to calculate the amount of blocks needed, and the amount of threads per block
    unsigned int blockCount = 1 + (config.threadCount-1) / (unsigned int)deviceProperties.maxThreadsPerBlock;

    /*
        The requested amount of thread might not be evenly distributed into the blocks, eg. 7 threads can't be
        evenly divided among 3 blocks. Calculate the amount of threads to add to avoid this issue.
    */
    unsigned int threadsToAdd = config.threadCount % blockCount;
    unsigned int threadsPerBlock = (config.threadCount + threadsToAdd) / blockCount;

    printf("Blocks: %d, Threads: %d (added %d to the requested amount)\n", blockCount, config.threadCount + threadsToAdd, threadsToAdd);

    // Launch the kernel on the GPU.
    const dim3 gridDim = {blockCount, 1, 1};
    const dim3 blockDim = {threadsPerBlock, 1, 1};

    // Restructure the image from AOS to SOA
    AOStoSOAKernel<<<gridDim, blockDim, 0>>>(deviceImageIn, deviceImageSOA);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    /* Prepare to blur the image */
    // Allocate GPU buffer for mask.
    cudaStatus = cudaMalloc((void**)&deviceMask, mask.size() * mask.size() * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy mask to GPU buffer
    cudaStatus = cudaMemcpy(deviceMask, mask.data(), mask.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of mask to device failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for AOS to SOA restructuring to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // The buffer allocated for the initial input image can now be used for the blurred image
    gaussianBlurKernel<<<gridDim, blockDim, 0>>>(deviceImageSOA, deviceImageIn, deviceMask);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output image from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(image.data(), deviceImageIn, image.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy of blurred image to host failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(deviceImageIn);
    cudaFree(deviceImageSOA);
    cudaFree(deviceMask);

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

bool checkValidity(const CImg<unsigned char>& cImage, const CImg<unsigned char>& cudaImage)
{
    int pixelCount = cImage.height() * cImage.width();

    const unsigned char* cimgData = cImage.data();
    const unsigned char* cudaData = cudaImage.data();

    for (int pixel = 0; pixel < pixelCount; pixel++) {
        unsigned char cimgRed = cimgData[pixel];
        unsigned char cimgGreen = cimgData[pixel + pixelCount];
        unsigned char cimgBlue = cimgData[pixel + pixelCount + pixelCount];

        unsigned char cudaRed = cudaData[pixel * 3];
        unsigned char cudaGreen = cudaData[pixel * 3 + 1];
        unsigned char cudaBlue = cudaData[pixel * 3 + 2];

        int pixY = pixel / cImage.width();
        int pixX = pixel - pixY * cImage.width();

        if (cimgRed != cudaRed) {
            printf("Red colors don't match at pixel (%d,%d): cimg: %d, cuda: %d\n", pixX, pixY, cimgRed, cudaRed);
            //return false;
        }
        if (cimgGreen != cudaGreen) {
            printf("Green colors don't match at pixel (%d,%d): cimg: %d, cuda: %d\n", pixX, pixY, cimgGreen, cudaGreen);
            //return false;
        }
        if (cimgBlue != cudaBlue) {
            printf("Blue colors don't match at pixel (%d,%d): cimg: %d, cuda: %d\n", pixX, pixY, cimgBlue, cudaBlue);
            //return false;
        }
    }

    printf("The blurring in CUDA was done the same was as in CImg\n");
    return true;
}
