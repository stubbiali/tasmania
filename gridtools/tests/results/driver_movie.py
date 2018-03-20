"""
Description: create a Matplotlib animation starting from Earth system science snapshots.

Usage: python driver_movie.py -i <infile> -a <argument> [-p <projection> -f <format> -o <outfile>]
       where 
           <infile> is the input file, 
           <argument> specifies what should be plotted,
           <projection> specifies the projection to use,
           <format> specifies the format for the movie,
           <outfile> is the output file name, without the extension.
       All the supported options for <argument>, <projection> and <format> are listed in ess_animation.py.
"""
import os, sys
import getopt
sys.path.append("..")
from ess_animation import ess_movie_maker

#
# Read arguments from command line
#
try:
	opts, args = getopt.getopt(sys.argv[1:], "hi:a:p:f:")
except getopt.GetoptError:
	print('Usage: driver_movie.py -i <infile> -a <argument> -p <projection> -f <format>')
	sys.exit(2)

infile = to_plot = projection = movie_format = outfile = None
do_help = False
for opt, arg in opts:
	if opt == '-i':
		infile = str(arg)
	elif opt == '-a':
		to_plot = str(arg)
	elif opt == '-p':
		projection = str(arg)
	elif opt == '-f':	
		movie_format = str(arg)
	elif opt == '-o':
		outfile = str(arg)
	elif opt == '-h':
		do_help = True

if do_help or len(sys.argv) < 2:
	print(__doc__)
	sys.exit()

assert (infile is not None and to_plot is not None), \
	   "Missing mandatory command line argument"

#
# Make movie
#
kwarg = {"file_name": infile, "to_plot": to_plot}
if projection is not None:
	kwarg["projection"] = projection
if movie_format is not None:
	kwarg["movie_format"] = movie_format
if outfile is not None:
	kwarg["out_file_name"] = outfile
ess_movie_maker(**kwarg)

