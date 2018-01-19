import os
import iris
iris.FUTURE.netcdf_promote = True
import logging
logger = logging.getLogger('sea_refactor')
logger.setLevel(logging.DEBUG)
from functools import lru_cache

# TODO, cant use lru cache as dict not hashable

def get_cube(bucket, conf, fcast_time):
    logger.debug("get_cube: %s, %s, %s" , bucket, conf, fcast_time)
    base = '/s3/{bucket}/model_data/'.format(bucket=bucket)
    filename = 'SEA_{conf}_{fct}.nc'.format(conf=conf,
                                         fct=fcast_time)
    path = os.path.join(base,filename)
    return iris.load(os.path.join(base, filename))


def get_slice(cube_list, var, extent=None):
    logger.debug("get_slice: %s, %s, %s" ,cube_list, var, extent)
    cube_list = cube_list.extract(var)
    logger.debug('get_slice. cube_list: %s', cube_list)
    assert len(cube_list) == 1
    cube = cube_list[0]
    cube = cube[-1]
    logger.debug("get_slice. cube: %s", cube)
    if extent:
        cube = cube.intersection(latitude=extent['latitude'], longitude=extent['longitude'])
    return cube

@lru_cache(32)
def get_data(bucket, conf, variable, t_fcst, lat_extent, lon_extent):
    cube = get_cube(bucket, conf, t_fcst)
    cube2d = get_slice(cube, variable, {'latitude':lat_extent, 'longitude':lon_extent})
    return cube2d