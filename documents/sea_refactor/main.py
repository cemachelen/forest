
import logging
import sys
import data_access as da
import vis
import bokeh.plotting
from bokeh.layouts import widgetbox, Row, Column
from bokeh.models.widgets import RadioButtonGroup
from bokeh.events import PanEnd, MouseWheel, PlotEvent, SelectionGeometry
from functools import partial
from threading import Thread
from bokeh.document import without_document_lock
from uuid import uuid4 as uuid
from tornado import gen
import copy
from spinners import Spinner
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)



doc = bokeh.plotting.curdoc()

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
doc.app_state = state = {'left_model': 0, 'right_model':1, 'var':0,'time': 0,'region':0}




@gen.coroutine
def state_change(state_prop, val):
    print('state change', state_prop, val)
    doc.app_state[state_prop] = val
    return do_a_update()

@gen.coroutine  
def do_a_update():
    print('update')
    spinner.show()
    update_id = uuid().hex
    doc.app_state['pending_update'] = update_id
    data = yield executor.submit(get_plot_data, update_id, copy.deepcopy(doc.app_state))
    if data['update_id'] == doc.app_state['pending_update']: # TODO: Is this needed. could we get out of sync if multiple fast interactions?
        doc.add_next_tick_callback(partial(update_plots, data))


@without_document_lock
def get_plot_data(update_id, state):
    result = {}
    for model_key in ['left_model', 'right_model']:
        model = models[state[model_key]]
        region = regions[state['region']]
        t_fcst = fcast_times[state['time']]
        cube2d = da.get_data(model['bucket'], model['conf'], var_names[state['var']], 
                            t_fcst, region['latitude'], region['longitude'])
        result[model_key] = cube2d
    result['update_id'] = update_id    
    return result

def update_plots(data):
    left_vis.update_plot(data['left_model'])
    right_vis.update_plot(data['right_model'])
    spinner.hide()

# Build the UI
spinner = Spinner()
spinner.hide()
vars_select = RadioButtonGroup(labels=var_names, active=state['var'])
vars_select.on_click(partial(state_change, 'var'))

region_select = RadioButtonGroup(labels=[r['name'] for r in regions], active=state['region'])
region_select.on_click(partial(state_change, 'region'))

left_model_select = RadioButtonGroup(labels=[m['name'] for m in models], active=state['left_model'])
left_model_select.on_click(partial(state_change, 'left_model'))

right_model_select = RadioButtonGroup(labels=[m['name'] for m in models], active=state['right_model'])
right_model_select.on_click(partial(state_change, 'right_model'))


left_vis = vis.CubePlot(width=800, height=600)
right_vis = vis.CubePlot(width=800, height=600, x_range=left_vis.x_range, y_range=left_vis.y_range)


do_a_update()
doc.add_root(
    Column(
        Row(
                spinner,
                vars_select,
                region_select
                
        ),
        Row( 
            Column(left_model_select, left_vis),
            Column(right_model_select, right_vis)
        )
    )
)