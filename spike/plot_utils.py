def deepcopy_Line2D(line):
	"""
	Return a deep copy of a :class:`matplotlib.lines.Line2D` object.

	Parameters
	----------
	line : obj
		The :class:`matplotlib.lines.Line2D` to copy.

	Return
	------
	obj :
		A deep copy of the input :class:`matplotlib.lines.Line2D`.
	"""
	assert type(line) == Line2D

	xdata = line.get_xdata()
	ydata = line.get_ydata()

	out_line = Line2D(xdata, ydata)
	out_line.update_from(line)

	return out_line

