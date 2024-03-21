extern "C" {

#define TILE_WIDTH              (32)
#define KIRSCH_NUM_DIRS         (8)
#define KIRSCH_RADIUS           (1)
#define KIRSCH_WIDTH            (2*KIRSCH_RADIUS+1)
#define NUM_OUTPUT_CMAPS        (3)
#define NUM_OUTPUT_CHANNELS     (3)

__constant__ unsigned int CMAPS[NUM_OUTPUT_CMAPS][KIRSCH_NUM_DIRS][NUM_OUTPUT_CHANNELS];
__constant__ int KF[KIRSCH_NUM_DIRS][KIRSCH_WIDTH][KIRSCH_WIDTH];

__global__
void kirsch_filter(
    unsigned char * const I, unsigned char * const O,
    unsigned int const width, unsigned int height,
    unsigned int const thres,
    unsigned int const cmap,
    unsigned int const scale)
{
    int const row = blockDim.y*blockIdx.y + threadIdx.y;
    int const col = blockDim.x*blockIdx.x + threadIdx.x;

    if ((row >= height) || (col >= width)) return;

    /* Load input image tile into shared memory */
    __shared__ unsigned int I_s[TILE_WIDTH][TILE_WIDTH];
    I_s[threadIdx.y][threadIdx.x] = I[row*width + col];
    __syncthreads();

    /* Skip outer edge */
    if ((row < KIRSCH_RADIUS) || (row >= height-KIRSCH_RADIUS)
            || (col < KIRSCH_RADIUS) || (col >= width-KIRSCH_RADIUS))
        return;

    /* Compute directional derivatives, threshold and max direction */
    int max_deriv_d = -1;
    int max_deriv = thres;
    for (int d = 0; d < KIRSCH_NUM_DIRS; d++) {
        int deriv = 0;
        for (int i = -KIRSCH_RADIUS; i <= KIRSCH_RADIUS; i++) {
            for (int j = -KIRSCH_RADIUS; j <= KIRSCH_RADIUS; j++) {
                int const iy = threadIdx.y + i;
                int const jx = threadIdx.x + j;
                if ((iy >= 0) && (iy < TILE_WIDTH) &&
                        (jx >= 0) && (jx < TILE_WIDTH)) {
                    deriv += KF[d][i+KIRSCH_RADIUS][j+KIRSCH_RADIUS]
                                * I_s[iy][jx];
                } else {
                    deriv += KF[d][i+KIRSCH_RADIUS][j+KIRSCH_RADIUS]
                                * I[(row+i)*width+(col+i)];
                }
            }
        }

        if ((deriv > thres) && (deriv > max_deriv)) {
            max_deriv_d = d;
            max_deriv = deriv;
        }
    }

    /* Write output colour for max direction */
    if (max_deriv_d < 0) return;

    for (int c = 0; c < NUM_OUTPUT_CHANNELS; c++) {
        for (int sy = 0; sy < scale; sy++) {
            for (int sx = 0; sx < scale; sx++) {
                int const pos = ((row*scale+sy)*width*scale+(col*scale+sx))
                                    *NUM_OUTPUT_CHANNELS + c;
                O[pos] = CMAPS[cmap][max_deriv_d][c];
            }
        }
    }
}

} /* extern "C" */
