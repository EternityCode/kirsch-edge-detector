#!/usr/bin/env python3

import argparse
import os
import sys
from PIL import Image

# Input Arguments
def sposint(val):
    val = int(val)
    if val <= 0:
        raise argparse.ArgumentTypeError(
            '{} invalid positive int value'.format(val))
    return val

aparser = argparse.ArgumentParser(
    description='Compute a map of edges and their directions from input ' \
                'images using the Kirsch operator.')
aparser.add_argument('-s', '--suffix', dest='img_suffix', metavar='suffix',
    default='_edge', type=str,
    help='a string to append to the end of the input filename')
aparser.add_argument('-c', '--colour', dest='img_colour_map',
    choices=('mono', 'sim', 'fpga'), default='sim',
    help='select the output edge colour mapping, \'sim\' and \'fpga\' are ' \
        'the ECE 327 colour mappings')
aparser.add_argument('-r', '--resize', dest='img_ratio', metavar='ratio',
    default=1, type=sposint, help='scaling factor of each input pixel')
aparser.add_argument('img_files', nargs='+', type=str)
args = aparser.parse_args()

# Colour Mappings
bg_colour = (0, 0, 0)
sim_colours = ((0, 100, 200), (100, 200, 0), (0, 200, 100), (255, 0, 0),
                (0, 0, 255), (200, 100, 0), (0, 255, 0), (200, 0, 100))
mono_colour = (255, 127, 0)
fpga_colours = ((0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
                (85, 85, 85), (255, 0, 255), (0, 255, 255), (255, 255, 255))

# Function Definitions
def getDerivatives(conv_table):
    conv_mask = [5, -3, -3, -3, -3, -3, 5, 5]
    rot = lambda l, n: l[-n:] + l[:-n]
    derivs = []
    for _ in range(8):
        derivs.append(sum([a * b for a,b in zip(conv_table, conv_mask)]))
        conv_mask = rot(conv_mask, 1)
    return derivs

def getEdgeColour(index, colour_map):
    if colour_map == 'sim':
        return sim_colours[index]
    elif colour_map == 'fpga':
        return fpga_colours[index]
    else:
        return mono_colour

# Main Program
for n, file in enumerate(args.img_files):
    try:
        img_grey = Image.open(args.img_files[0]).convert('L')
    except IOError:
        msg = 'Fatal Error: Could not open file \'{name}\'.'.format(name=file)
        sys.exit(msg)

    img_edge = Image.new(
        'RGB',
        (img_grey.width * args.img_ratio, img_grey.height * args.img_ratio),
        bg_colour)

    for y in range(1, img_grey.height - 1):
        msg = "\r[{:0>3d}/{:0>3d}] Processing File: {} (Row {:0>5d} of {:0>5d})"
        sys.stdout.write(
            msg.format(n + 1, len(args.img_files),
                        file, y + 1, img_grey.height - 1)
        )
        sys.stdout.flush()

        for x in range(1, img_grey.width - 1):
            derivs = getDerivatives([img_grey.getpixel((x - 1, y - 1)),
                                    img_grey.getpixel((x, y - 1)),
                                    img_grey.getpixel((x + 1, y - 1)),
                                    img_grey.getpixel((x + 1, y)),
                                    img_grey.getpixel((x + 1, y + 1)),
                                    img_grey.getpixel((x, y + 1)),
                                    img_grey.getpixel((x - 1, y + 1)),
                                    img_grey.getpixel((x - 1, y))])
            if max(derivs) > 383:
                pos = next(pos for pos in range(len(derivs))
                            if derivs[pos] == max(derivs))
                for i_x in range(args.img_ratio):
                    for i_y in range(args.img_ratio):
                        img_edge.putpixel(
                            (x * args.img_ratio + i_x,
                                y * args.img_ratio + i_y),
                            getEdgeColour(pos, args.img_colour_map))

    try:
        img_edge.save('{name}{suff}{ext}'.format(
            name=os.path.splitext(file)[0], suff=args.img_suffix,
            ext=os.path.splitext(file)[1]))
    except IOError:
        msg = 'Fatal Error: Could not write to file ' \
                '\'{name}\'.'.format(name=file)
        sys.exit(msg)

    print('')

print('Success.')

