
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot
import matplotlib.cm

import cartopy
import cartopy.crs
import cartopy.io.img_tiles
import logging

import numpy

import bokeh.plotting

import random

logger = logging.getLogger('sea_refactor')
logger.setLevel(logging.DEBUG)


class CubePlot(bokeh.plotting.Figure):
    
    __implementation__ = """
    import {Plot} from "models/plots/plot"
    export class CubePlot extends Plot
        type: "CubePlot"
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__source = bokeh.plotting.ColumnDataSource(data=dict(x=[],y=[],dw=[],dh=[],image=[]))
        self.image_rgba(image='image',source=self.__source,x='x',y='y',dw='dw',dh='dh')
    
    def update_plot(self, cube2d):
        img = make_plot_img(cube2d)
        x = cube2d.coords('longitude')[0].points
        y = cube2d.coords('latitude')[0].points
        logger.debug("x = %s - %s, y= %s - %s", x[0], x[-1], y[0], y[-1])


        self.__source.data = dict(
            x=[x[0]],
            y=[y[0]],
            dw=[x[0] - x[-1]],
            dh=[y[0] - y[-1]],
            image=[img])
        self.x_range.start = x[0]
        self.x_range.end = x[-1]
        self.y_range.start = y[0]
        self.y_range.end = y[-1]



def make_plot_img(cube2d):
    fig = matplotlib.pyplot.figure("My Fig", figsize=(4.0,3.0))
    fig.clf()
    current_axes = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
    current_axes.set_position([0, 0, 1, 1])

    main_plot = current_axes.pcolormesh(
                                     cube2d.coords('longitude')[0].points, 
                                     cube2d.coords('latitude')[0].points, 
                                     cube2d.data,
                                     edgecolors="none")

    coastline_50m = cartopy.feature.NaturalEarthFeature('physical', 
                                                            'coastline', 
                                                            '50m', 
                                                            edgecolor='0.5', 
                                                            facecolor = 'none')
    current_axes.add_feature(coastline_50m)    

    fig.canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    
    # Get the RGB buffer from the figure
    h, w = fig.canvas.get_width_height()
    # logger.log('make_plot_image  width={0}\nheight={1}'.format(w,h))
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    buf = numpy.flip(buf,axis=0)
    return buf