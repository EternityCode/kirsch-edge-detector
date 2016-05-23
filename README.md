# Kirsch Edge Detector
Compute a map of edges and their directions (dark to bright) from input images using the Kirsch operator.

This script is written in Python, and supports colour and greyscaled images of arbitrary dimensions, it is able to process any image format that Pillow supports. This is an implementation of an algorithm in the public domain, its structure has no functional resemblance to the University of Waterloo ECE 327 lab project (which is written in VHDL to describe hardware), it is therefore not subject to Policy 71.

## Dependencies
- Python 3
- Requires Pillow (fork of PIL): https://github.com/python-pillow/Pillow
- PyOpenCL (optional)

Note: GPU Acceleration w/ OpenCL is not fully-implemented yet.

## Help Output
```
usage: kirsch.py [-h] [-s suffix] [-a] [-c {mono,sim,fpga}]
                 [-t deriv_threshold] [-r ratio]
                 img_files [img_files ...]

Compute a map of edges and their directions from input images using the Kirsch
operator.

positional arguments:
  img_files

optional arguments:
  -h, --help            show this help message and exit
  -s suffix, --suffix suffix
                        a string to append to the end of the input filename
  -a, --accel-gpu       enable GPU acceleration through OpenCL
  -c {mono,sim,fpga}, --colour {mono,sim,fpga}
                        select the output edge colour mapping, 'sim' and
                        'fpga' are the ECE 327 colour mappings (default: sim)
  -t deriv_threshold, --threshold deriv_threshold
                        the maximum edge direction derivative threshold
                        (default: 383)
  -r ratio, --resize ratio
                        scaling factor of each input pixel (default: 1)
```
