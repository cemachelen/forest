"""
Zooming and enhancing images in bokeh can be done by rendering
a coarse image for the zoomed out extent and a collection
of high resolution patches as the axis extents get shorter

Careful management of both the number and reuse of the high
resolution patches is important for memory/performance reasons.
If memory were not an issue one could just render the highest
resolution image across the full domain

A first in first out (FIFO) cache can be used to manage
the high resolution patches. The most stale patch can
be replaced by the most current patch, thus keeping the
number of patches constant.

If a zoom view is completely inside a previous patch, then that
patch is good enough for the current view and no extra work
or memory is needed
"""


def boxes_overlap(box_1, box_2):
    """Decide if two boxes overlap

    Two boxes overlap if their lower left corner
    is contained in the lowest left-most box of
    the two

    :param box_1: tuple of x, y, dw, dh representing a box
    :param box_2: tuple of x, y, dw, dh representing a box
    :returns: True if boxes overlap
    """
    x1, y1, dw1, dh1 = box_1
    x2, y2, dw2, dh2 = box_2
    if x1 < x2:
        dw = dw1
    else:
        dw = dw2
    if y1 < y2:
        dh = dh1
    else:
        dh = dh2
    return (abs(x1 - x2) < dw) & (abs(y1 - y2) < dh)
