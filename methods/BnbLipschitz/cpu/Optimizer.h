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