
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot
import matplotlib.cm

import cartopy
import cartopy.crs
import cartopy.io.img_tiles
import logging
import datetime
import numpy

import bokeh.plotting

import random
from functools import partial
from bokeh.events import PanEnd #TODO: Also need MouseWheel to capture zoom.

from functools import lru_cache


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

        self.on_event(PanEnd, self.resize)
        self.__orig_cube = None

    def resize(self, event):
        logger.debug("resize: %s, %s", event.x, event.y)
        logger.debug("resize range, scale, axis: %s, %s, %s", self.x_range, self.x_scale, self.xaxis)
        if self.__orig_cube:
            new_cube = self.__orig_cube.extract(latitude=(), longitude=())
            self._update_plot_from_cube(new_cube)
        
        
    def update_plot(self, cube2d):
        self.__orig_cube = cube2d
        self._update_plot_from_cube(cube2d)

    def _update_plot_from_cube(self, cube2d):
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
        if self._document:
            self._document.add_next_tick_callback(partial(self._update_range,x[0],x[-1],y[0],y[-1]))

    def _update_range(self,x_start, x_end, y_start, y_end):
        self.x_range.start = x_start
        self.x_range.end = x_end
        self.y_range.start = y_start
        self.y_range.end = y_end


@lru_cache(32) # TODO: are reling on the fact that cube objects are being cached or are we hashing on metadata?
def make_plot_img(cube2d):
    s = datetime.datetime.now()
    fig = matplotlib.pyplot.figure("My Fig", figsize=(4.0,3.0), dpi=300)
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
    logger.info('make_plot_img took %s' % (datetime.datetime.now() - s))
    return buf