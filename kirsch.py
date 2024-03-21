#!/usr/bin/env python3

import argparse
import os
import sys
import time
import math

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cupy as cp

# Input Arguments
def sposint(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentTypeError(f'{val} invalid positive int value')
    return val

aparser = argparse.ArgumentParser(
    description='Compute a map of edges and their directions from input ' \
                'images using the Kirsch operator.')
aparser.add_argument('-s', '--suffix', dest='img_suffix', metavar='suffix',
    default='_edge', type=str,
    help='a string to append to the end of the input filename')
aparser.add_argument('-a', '--accel-gpu', dest='accel_gpu', default=False,
    action='store_true',
    help='enable GPGPU acceleration through CUDA')
aparser.add_argument('-c', '--colour', dest='img_colour_map',
    choices=('mono', 'sim', 'fpga'), default='sim', type=str,
    help='select the output edge colour mapping, \'sim\' and \'fpga\' are ' \
        'the ECE 327 colour mappings (default: sim)')
aparser.add_argument('-t', '--threshold', dest='threshold',
    metavar='deriv_threshold', default='383', type=int,
    help='the maximum edge direction derivative threshold (default: 383)')
aparser.add_argument('-r', '--resize', dest='img_ratio', metavar='ratio',
    default=1, type=sposint,
    help='scaling factor of each input pixel (default: 1)')
aparser.add_argument('img_files', nargs='+', type=str)

args = aparser.parse_args()

# Directional Colour Mappings
bg_colour = np.array([0, 0, 0])
cmaps = np.array([
    # sim colours
    [[0, 100, 200], [100, 200, 0], [0, 200, 100], [255, 0, 0],
     [0, 0, 255], [200, 100, 0], [0, 255, 0], [200, 0, 100]],
    # fpga colours
    [[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0],
     [85, 85, 85], [255, 0, 255], [0, 255, 255], [255, 255, 255]],
    # mono colour
    [[255, 127, 0], [255, 127, 0], [255, 127, 0], [255, 127, 0],
      [255, 127, 0], [255, 127, 0], [255, 127, 0], [255, 127, 0]]
    ])

# Function Definitions
def getKirschFilters():
    kirsch = [5, -3, -3, -3, -3, -3, 5, 5]
    rot = lambda l, n: l[-n:] + l[:-n]
    filts = np.zeros((8, 3, 3), dtype=np.int32)
    for d in range(8):
        filts[d] = np.array([kirsch[0:3],
                             [kirsch[7], 0, kirsch[3]],
                             kirsch[6:3:-1]], dtype=np.int32)
        kirsch = rot(kirsch, 1)
    return filts

# Main Program
def main():
    cmap = args.img_colour_map
    cmap_idx = 0 if (cmap == 'sim') else (1 if (cmap == 'fpga') else 2)
    thres = args.threshold
    scale = args.img_ratio
    filts = getKirschFilters()

    for n, file in enumerate(args.img_files):
        t0 = time.time_ns()

        try:
            img_grey = np.asarray(Image.open(args.img_files[n]).convert('L'))
        except IOError:
            sys.exit(f'Fatal Error: Could not open file {file}')

        height, width = img_grey.shape

        print(f'[File {n+1} of {len(args.img_files)}: ' \
              f'{file} ({width}x{height})]')

        if not(args.accel_gpu):
            img_edge = np.zeros((height*scale, width*scale, 3), dtype=np.uint8)

            # Compute directional derivative,
            ct0 = time.time_ns()
            derivs = np.zeros((8, height, width), dtype=np.int32)
            for d in range(8):
                filt = np.flipud(np.fliplr(filts[d]))
                derivs[d] = convolve2d(img_grey, filt, mode='same',
                                       boundary='symm')

            # Threshold and max direction
            max_derivs = np.max(derivs, axis=0)
            max_dirs = np.argmax(derivs, axis=0)

            for irow in range(height):
                for icol in range(width):
                    def getEdgePixel(dv, dr):
                        return cmaps[cmap_idx][dr] if dv > thres else bg_colour
                    pixVal = getEdgePixel(max_derivs[irow, icol],
                                          max_dirs[irow, icol])
                    for oy in range(scale):
                        for ox in range(scale):
                            orow = irow*scale + oy
                            ocol = icol*scale + ox
                            img_edge[orow][ocol] = pixVal

            ct1 = time.time_ns()
            print(f'CPU time: {(ct1-ct0)/1e6} ms')

            img_edge = Image.fromarray(img_edge)

        else:
            d_img_grey = cp.asarray(img_grey, dtype=np.uint8)
            d_img_edge = cp.asarray(np.zeros((height*scale, width*scale, 3),
                                             dtype=np.uint8))

            with open('kirsch.cu', 'r') as cu_src:
                kirsch_mod = cp.RawModule(code=cu_src.read())

            # Constants
            pd_KF = kirsch_mod.get_global('KF')
            d_KF = cp.ndarray(filts.shape, cp.int32, pd_KF)
            d_KF[::] = cp.asarray(filts, cp.int32)

            pd_CMAPS = kirsch_mod.get_global('CMAPS')
            d_CMAPS = cp.ndarray(cmaps.shape, cp.uint32, pd_CMAPS)
            d_CMAPS[::] = cp.asarray(cmaps, cp.uint32)

            # Execute kernel
            kirsch_filter = kirsch_mod.get_function('kirsch_filter')
            kt0 = time.time_ns()
            kirsch_filter((math.ceil(width/32), math.ceil(height/32)),
                            (32,32,),
                            (d_img_grey, d_img_edge, width, height,
                            thres, cmap_idx, scale))
            kt1 = time.time_ns()
            print(f'CUDA Kernel time: {(kt1-kt0)/1e6} ms')

            img_edge = Image.fromarray(cp.asnumpy(d_img_edge))

        try:
            ofile = f'{os.path.splitext(file)[0]}{args.img_suffix}' \
                    f'{os.path.splitext(file)[1]}'
            owidth, oheight = img_edge.size
            img_edge.save(ofile)
            print(f'Output: {ofile} ({owidth}x{oheight})')
        except IOError:
            sys.exit(f'Fatal Error: Could not write to file {file}')

        t1 = time.time_ns()
        print(f'Total file time: {(t1-t0)/1e6} ms')

        print('')

    print('Success.')

if __name__ == '__main__': main()
