# build_tranforms

from .add_transformation_classes import transformation_class

# from .transformations import *

from .transformations.utils import *
from .transformations.basic import *
from .transformations.spatial_correction import *
from .transformations.clustering import *
from .transformations.open_files import *
from .transformations.custom import *
from os.path import dirname, join

import json


def build_transforms(args):
    transformations = transformation_class()
    filepath = args['transforms_path'][0]
    print(filepath)
    with open(filepath) as f:
        transform_config = json.load(f)
    print(transform_config)
    script_dir = dirname(__file__)  # <-- absolute dir the script is in
    abs_file_path = join(script_dir, 'transformation_dispatch.cfg')
    with open(abs_file_path) as f:
        dispatch = eval(f.read())
    # dispatch = {
    #         'log_transform': log_transform,
    #         'save_to_csv': save_to_csv,
    #         'find_clusters': find_clusters,
    #         'exclude_clustered_data': exclude_clustered_data,
    #         'shift_baseline': shift_baseline,
    #         'subtract_columns_class': subtract_columns_class,
    #         'merge_data_frames': merge_data_frames,
    #         'median_group_by': median_group_by,
    #         'melt_class': melt_class,
    #         'rolling_median': rolling_median,
    #         'open_files': open_files,
    #         'merge_corresponding_files': merge_corresponding_files,
    #         'local_spatial_correction': local_spatial_correction,
    #         'large_area_spatial_correction': large_area_spatial_correction,
    #         'open_data_meta_files': open_data_meta_files}

    for transform_i in transform_config:
        if 'skip' not in transform_i:
            transform_i['skip'] = False
        print(transform_i['transformation_args'])
        transformations.add_transformation(name=transform_i['name'],
                                           order=transform_i['order'],
                                           skip=transform_i['skip'],
                                           transformation=dispatch[transform_i['transformation_func']](
                                               **transform_i['transformation_args']))
    return transformations
