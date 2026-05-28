#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"


// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1)
                {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}

// Function to perform convolution on input using kernel w
inline double *convolve2d(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    if (result != NULL && input != NULL && w != NULL)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}

// Function to evolve Lenia
double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums, const int myid, const int procs)
{

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",     /* file name */
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    int rows_per_proc = rows / procs;
    int my_start = myid * rows_per_proc;
    int my_end = my_start + rows_per_proc;
    // Allocate memory
    double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
    double *local_world = calloc(rows_per_proc * cols, sizeof(double));
    double *local_tmp   = calloc((rows_per_proc + 26) * cols, sizeof(double));

    // Generate convolution kernel
    w=generate_kernel(w,kernel_size);

    for (int o = 0; o < num_orbiums; o++) {
        if (orbiums[o].row >= my_start && orbiums[o].row < my_end) {
            local_world = place_orbium(local_world, rows_per_proc, cols, orbiums[o].row - my_start, orbiums[o].col, orbiums[o].angle);
        }
    }

    int neighbour_above = (myid - 1 + procs) % procs;
    int neighbour_below = (myid + 1) % procs;

    double *ghost_bot = (double *)calloc(13 * cols, sizeof(double));
    double *ghost_top = (double *)calloc(13 * cols, sizeof(double));

    double *include_ghost_world = calloc((rows_per_proc + 26) * cols, sizeof(double));

    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++)
    {

        // izmenjava ghost cells
        MPI_Sendrecv(
            &local_world[0],
            13 * cols,
            MPI_DOUBLE,
            neighbour_above, 0,
            &ghost_top[0],
            13 * cols,
            MPI_DOUBLE,
            neighbour_above, 1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            &local_world[(rows_per_proc - 13) * cols],
            13 * cols,
            MPI_DOUBLE,
            neighbour_below, 1, 
            &ghost_bot[0],
            13 * cols,
            MPI_DOUBLE,
            neighbour_below, 0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        memcpy(&include_ghost_world[0], ghost_top, 13*cols*sizeof(double));
        memcpy(&include_ghost_world[13*cols], local_world, rows_per_proc*cols*sizeof(double));
        memcpy(&include_ghost_world[(rows_per_proc + 13) * cols], ghost_bot, 13*cols*sizeof(double));
        // Convolution
        local_tmp = convolve2d(local_tmp, include_ghost_world, w, rows_per_proc + 26, cols, kernel_size, kernel_size);
        
        // Evolution
        for (unsigned int i = 0; i < rows_per_proc; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                local_world[i * cols + j] += dt * growth_lenia(local_tmp[(i + 13) * cols + j]);
                local_world[i * cols + j] = fmin(1, fmax(0, local_world[i * cols + j])); // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * cols + j] = (uint8_t)(local_world[i * cols + j] * 255);
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    free(w);
    free(local_tmp);
    free(ghost_top);
    free(ghost_bot);
    free(include_ghost_world);
    return local_world;
}
