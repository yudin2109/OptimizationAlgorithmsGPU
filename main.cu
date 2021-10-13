%%cu
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

__device__ double error = 0.001;

#define sqr(x) ((x) * (x))

// Compares *address and val, sets minimum to *address. Works thread-safely 
// Well known hack since CUDA have no native realisation for atomicMin(double*, double)
// Source https://stackoverflow.com/questions/55140908/can-anybody-help-me-with-atomicmin-function-syntax-for-cuda
__device__
double atomicMin(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(min(val,
                               __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__
double atomicMax(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(val,
                               __longlong_as_double(assumed))));
    } while (assumed != old);

    return __longlong_as_double(old);
}

double funcHost(double* x) {
    return -20 * exp(-1 * sqrt((sqr(x[0]) + sqr(x[1])) / 2) / 5)
        - exp((cos(2 * M_PI * x[0]) + cos(2 * M_PI * x[1])) / 2)
        + 20 + M_E;
}

__device__
double f(double* x) {
    return -20 * exp(-1 * sqrt((sqr(x[0]) + sqr(x[1])) / 2) / 5)
        - exp((cos(2 * M_PI * x[0]) + cos(2 * M_PI * x[1])) / 2)
        + 20 + M_E;
}

__device__
double calcRobustnessCoeff(double x) {
    return exp(x);
}

__device__
bool stopCriteria(
    double localLowerBound,
    double* globalResult
) {
    return localLowerBound + error >= *globalResult;
}

__device__
double getLipschitzConstantLocal(
    int nDims,
    double* xStart,
    double* xAround,
    double* gridSteps
) {
    double LipschitzConstantMax = 0;

    // Initialize the neighbour node
    for (int i = 0; i < nDims; ++i) {
        xAround[i] = xStart[i];
    }

    // Going around of xStart to estimate Lipschitz constant
    for (int i = 0; i < nDims; ++i) {
        xAround[i] += gridSteps[i];
        LipschitzConstantMax = max(LipschitzConstantMax,
                                    fabs(f(xStart) - f(xAround)) / gridSteps[i]);
        xAround[i] -= gridSteps[i];
    }

    return LipschitzConstantMax;
}


// Calculates minimum and Lipschitz constant for hyperinterval
// by searching on grid
__global__
void gridKernel(
    int nDims,
    double** borders,
    int* blocksPartition,
    int threadsPerBlock,
    int threadsPerDimensionInBlock,
    double* gridSteps,
    double error,
    double* globalResult,
    double* LipschitzConstant
) {
    int blockIndex = blockIdx.x;
    int threadIndex = threadIdx.x;
    printf("%d\n", blockIndex);
   
    int dimensionIndex;
    double x[5];
    double xAround[5];

    for (int i = 0; i < nDims; ++i) {
        dimensionIndex = (blockIndex % blocksPartition[i]) * threadsPerDimensionInBlock
                            + threadIndex % threadsPerDimensionInBlock;
        x[i] = dimensionIndex * gridSteps[i] + borders[i][0];
        threadIndex /= threadsPerDimensionInBlock;
        blockIndex /= blocksPartition[i];
    }

    double FuncValue = f(x);
    printf("%f\n", FuncValue);
    atomicMin(globalResult, FuncValue);
    double LipschitzConstantLocal
                = getLipschitzConstantLocal(
                                nDims,
                                x,
                                xAround,
                                gridSteps);
    atomicMax(LipschitzConstant, LipschitzConstantLocal);
}

void findMinCuda(
    int nDims,
    double** borders,
    double* minValue,
    double* LipschitzConstant
) {
    double** bordersOnDevice;
    cudaMallocManaged(&bordersOnDevice, nDims * sizeof(double*));
    for (int i = 0; i < nDims; ++i) {
        cudaMallocManaged(&bordersOnDevice[i], sizeof(double) * 2);
        cudaMemcpy(bordersOnDevice[i], borders[i], sizeof(double) * 2, cudaMemcpyHostToDevice);
    }

    int* blocksPartition;
    cudaMallocManaged(&blocksPartition, nDims * sizeof(int));
    for (int i = 0; i < nDims; ++i) {
        blocksPartition[i] = 8;
    }

    int threadsPerDimensionInBlock = 16;
    int threadsPerBlock = 1;
    for (int i = 0; i < nDims; ++i) {
        threadsPerBlock *= threadsPerDimensionInBlock;
    }

    double* gridStepsHost = new double[nDims];
    for (int i = 0; i < nDims; ++i) {
        gridStepsHost[i] = (borders[i][1] - borders[i][0])
            / (8 * threadsPerDimensionInBlock);
    }

    double* gridStepsDevice;
    cudaMallocManaged(&gridStepsDevice, nDims * sizeof(double));
    cudaMemcpy(gridStepsDevice, gridStepsHost, sizeof(double) * nDims, cudaMemcpyHostToDevice);


    double* minValueDevice;
    cudaMallocManaged(&minValueDevice, sizeof(double));
    double* LipschitzConstantDevice;
    cudaMallocManaged(&LipschitzConstantDevice, sizeof(double));
    double defaultMinValue = 0;
    double defaultLCValue = 0;
    cudaMemcpy(minValueDevice, &defaultMinValue, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(LipschitzConstantDevice, &defaultLCValue, sizeof(double), cudaMemcpyHostToDevice);

    gridKernel<<<64, threadsPerBlock>>>(
        nDims,
        bordersOnDevice,
        blocksPartition,
        threadsPerBlock,
        threadsPerDimensionInBlock,
        gridStepsDevice,
        error,
        minValueDevice,
        LipschitzConstantDevice
    );
    cudaMemcpy(minValue, minValueDevice, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(LipschitzConstant, LipschitzConstantDevice, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Found minimum = " << *minValue << std::endl;
}

void findMinHost(
    int nDims,
    double** borders,
    double* minValue,
    double* LipschitzConstant
) {
    int nodesPerDimension = 8 * 16;
    int totalNodesCount = 1;
    for (int i = 0; i < nDims; ++i) {
        totalNodesCount *= nodesPerDimension;
    }

    double* x = new double[nDims];
    for (int i = 0; i < totalNodesCount; ++i) {
        int index = i;
        for (int d = 0; d < nDims; ++d) {
            x[d] = borders[d][0] + ((borders[d][1] - borders[d][0]) / nodesPerDimension)
                * (index % nodesPerDimension);
            index /= nodesPerDimension;
        }

        double funcValue = funcHost(x);
        *minValue = min(*minValue, funcValue);
    }
    std::cout << "Found minimum = " << *minValue << std::endl;
}

int main() {
    double** borders = new double*[2];
    borders[0] = new double[2];
    borders[1] = new double[2];

    borders[0][0] = borders[1][0] = -7;
    borders[0][1] = borders[1][1] = 7;

    double* result = new double[1];
    double* LC = new double[1];

    auto start = high_resolution_clock::now();
    findMinCuda(2, borders, result, LC);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time spent: " << duration.count() << endl;

    start = high_resolution_clock::now();
    findMinHost(2, borders, result, LC);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "Time spent: " << duration.count() << endl;
    
    return 0;
}