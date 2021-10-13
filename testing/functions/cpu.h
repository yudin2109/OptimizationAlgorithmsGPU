#include <math.h>
#define sqr(x) ((x) * (x))


namespace cpu_functions {

    double ackley2d(double* x) {
        return -20 * exp(-1 * sqrt((sqr(x[0]) + sqr(x[1])) / 2) / 5)
            - exp((cos(2 * M_PI * x[0]) + cos(2 * M_PI * x[1])) / 2)
            + 20 + M_E;
    }

    double ackley3d(double* x) {
        return -20 * exp(-1 * sqrt((sqr(x[0]) + sqr(x[1]) + sqr(x[2])) / 3) / 5)
            - exp((cos(2 * M_PI * x[0]) + cos(2 * M_PI * x[1]) + cos(2 * M_PI * x[2])) / 3)
            + 20 + M_E;
    }

    double ackleyGeneral(int n, double* x) {
        double sqr_sum = 0;
        double cos_sum = 0;
        for (int i = 0; i < n; ++i) {
            sqr_sum += sqr(x[i]);
            cos_sum += cos(2 * M_PI * x[i]);
        }
        return -20 * exp(-1 * sqrt(sqr_sum / n) / 5)
            - exp(cos_sum / n)
            + 20 + M_E;
    }

    double rosenbrock2d (double* x) {
        return sqr(x[0] - 1) + 100 * sqr(x[1] - sqr(x[0]));
    }

    double rosenbrock3d (double* x) {
        return sqr(x[0] - 1) + 100 * sqr(x[1] - sqr(x[0])) + 
            sqr(x[1] - 1) + 100 * sqr(x[2] - sqr(x[1]));
    }

    double rosenbrockGeneral(int n, double* x) {
        double sum = 0;
        for (int i = 0; i < n - 1; ++i) {
            sum += sqr(x[i] - 1) + 100 * sqr(x[i + 1] - sqr(x[i]));
        }
        return sum;
    }
} // namespace cpu_functions
