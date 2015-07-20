# Kirsch Edge Detector
Compute a map of edges and their directions (dark to bright) from input images using the Kirsch operator.

This script is written in Python, and supports colour and greyscaled images of arbitrary dimensions, it is able to process any image format that Pillow supports. This is an implementation of an algorithm in the public domain, its structure has no functional resemblance to the University of Waterloo ECE 327 lab project (which is written in VHDL to describe hardware), it is therefore not subject to Policy 71.

## Dependencies
- Python 3
- Requires Pillow (fork of PIL): https://github.com/python-pillow/Pillow

## Help Output
```
usage: kirsch.py [-h] [-s suffix] [-c {mono,sim,fpga}] [-r ratio]
                 img_files [img_files ...]

Compute a map of edges and their directions from input images using the Kirsch
operator.

positional arguments:
  img_files

optional arguments:
  -h, --help            show this help message and exit
  -s suffix, --suffix suffix
                        a string to append to the end of the input filename
  -c {mono,sim,fpga}, --colour {mono,sim,fpga}
                        select the output edge colour mapping, 'sim' and
                        'fpga' are the ECE 327 colour mappings
  -r ratio, --resize ratio
                        scaling factor of each input pixel
```
