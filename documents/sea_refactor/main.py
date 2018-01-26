
import logging
import sys
import data_access as da
import vis
import bokeh.plotting
from bokeh.layouts import widgetbox, Row, Column
from bokeh.models.widgets import RadioButtonGroup
from functools import partial
from bokeh.events import PanEnd, MouseWheel, PlotEvent, SelectionGeometry


logger = logging.getLogger('sea_refactor')
logger.setLevel(logging.DEBUG)


bucket_name = 'stephen-sea-public-london'
fcast_times = ['20180110T0000Z']
plot_names = ['air_temperature', 'x_wind', 'y_wind']
var_names = ['air_temperature', 'x_wind', 'y_wind']
regions = [
    {'name':'se_asia', 'latitude':(-18.0, 29.96), 'longitude':(90.0, 153.96)},
    {'name':'indonesia', 'latitude':(-15.1, 1.0865), 'longitude':(99.875, 120.111)}
]
models = [
    {'conf': 'n1280_ga6', 'name':'N1280 GA6 LAM Model', 'bucket':bucket_name},
    {'conf':'km4p4_ra1t', 'name':'SE Asia 4.4KM RA1-T ', 'bucket':bucket_name}
]


# Attach state to the current doc. The state dictates we plot.
bokeh.plotting.curdoc().app_state =  state = {'left_model': 0, 'right_model':1, 'var':0,'time': 0,'region':0}

def state_change(state_prop, val):
    state = bokeh.plotting.curdoc().app_state
    state[state_prop] = val
    update_plot_from_state(state)

def update_plot_from_state(state):
    update_plot(left_vis,'left_model', state)
    update_plot(right_vis,'right_model', state)

def update_plot(plot, model, state):
    model = models[state[model]]
    region = regions[state['region']]
    t_fcst = fcast_times[state['time']]
    cube2d = da.get_data(model['bucket'], model['conf'], var_names[state['var']], 
                        t_fcst, region['latitude'], region['longitude'])
    plot.update_plot(cube2d)    


    


# Build the UI

vars_select = RadioButtonGroup(labels=var_names, active=state['var'])
vars_select.on_click(partial(state_change, 'var'))

region_select = RadioButtonGroup(labels=[r['name'] for r in regions], active=state['region'])
region_select.on_click(partial(state_change, 'region'))

left_model_select = RadioButtonGroup(labels=[m['name'] for m in models], active=state['left_model'])
left_model_select.on_click(partial(state_change, 'left_model'))

right_model_select = RadioButtonGroup(labels=[m['name'] for m in models], active=state['right_model'])
right_model_select.on_click(partial(state_change, 'right_model'))


left_vis = vis.CubePlot()
right_vis = vis.CubePlot(x_range=left_vis.x_range, y_range=left_vis.y_range)



def callback(event):
    print('Python:Click, %r' % event)

update_plot_from_state(bokeh.plotting.curdoc().app_state)
bokeh.plotting.curdoc().add_root(
    Column(
        Row(
                vars_select,
                region_select
        ),
        Row( 
            Column(left_model_select, left_vis),
            Column(right_model_select, right_vis)
        )
    )
)