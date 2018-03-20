"""
This module contains the shape classes though which the user specifies
the domain of the output data that has to be computed.
"""


class Rectangle:
    def __init__(self, up_left, down_right):
        self.up_left = up_left
        self.down_right = down_right
