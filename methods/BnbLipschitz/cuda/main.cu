#include <iostream>
#include <cmath>

struct HyperInterval {
    int nDims;
    double* leftBorders = nullptr;
    double* rightBorders = nullptr;
    double upperBound, lowerBound;
};

typedef struct HyperInterval HyperInterval;

HyperInterval copyInterval(
    int nDims,
    double* _leftBorders,
    double* _rightBorders,
    double _upperBound,
    double _lowerBound
) {
    double* leftBorders = (double*) calloc(nDims, sizeof(double));
    double* rightBorders = (double*) calloc(nDims, sizeof(double));

    for (int i = 0; i < nDims; ++i) {
        leftBorders[i] = _leftBorders[i];
        rightBorders[i] = _rightBorders[i];
    }

    return (HyperInterval){
        .nDims = nDims,
        .leftBorders = leftBorders,
        .rightBorders = rightBorders,
        .upperBound = _upperBound,
        .lowerBound = _lowerBound
    };
}

void freeInterval(HyperInterval* interval) {
    free(interval->leftBorders);
    free(interval->rightBorders);
}

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
    int index;
    double x[5];
    double xAround[5];

    for (int i = 0; i < nDims; ++i) {
        int dimensionIndex = (blockIndex % blocksPartition[i]) * threadsPerDimensionInBlock
                            + threadIndex % threadsPerDimensionInBlock;
        x[i] = dimensionIndex * gridSteps[i] + borders[i][0];
        threadIndex /= threadsPerDimensionInBlock;
        blockIndex /= blocksPartition[i];
    }

    double FuncValue = f(x);
    atomicMin(globalResult, FuncValue);
    double LipschitzConstantLocal
                = getLipschitzConstantLocal(
                                dimensions,
                                x,
                                xAround,
                                gridSteps);
    atomicMin(LipschitzConstant, LipschitzConstantLocal);
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