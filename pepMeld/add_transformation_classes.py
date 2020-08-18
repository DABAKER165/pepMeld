"""Add Transformation Clases"""

from .transformations.utils import *
from .transformations.basic import *
from .transformations.spatial_correction import *
from .transformations.clustering import *
from .transformations.open_files import *
from .transformations.custom import *


class transformation_class(utils_transformations, basic_transformations, spatial_correction, clustering,
                           open_transform_files, custom):
    def __init__(self):
        self.transformation = {}
        self.names = {}
        self.skip = {}
        self.descriptor_columns = set()
        self.sample_name_column = "SAMPLE_NAME"
        self.vendor_name_column = "VENDOR_NAME"
        self.data_columns = []
        self.old_data_columns = []
        self.number_to_name_dict = {}
        self.df_dict = {}

    def add_transformation(self,
                           name=None,
                           order=None,
                           skip=False,
                           transformation=None):
        if name in self.names.keys():
            raise RuntimeError('Cant have duplicate Names: ' + str(name))
        if order in list(self.names.values()):
            raise RuntimeError('Cant have duplicate Order: ' + str(name) + '|' + str(order))
        if transformation is None:
            raise RuntimeError('Must have a transformation set')
        self.names[name] = order
        self.transformation[order] = transformation
        self.skip[order] = skip

    def remove_transformation(self,
                              name=None):
        if name is None:
            print("WARNING: Must have a name nothing removed")
            return
        if name not in self.names.keys():
            print("WARNING: Name not in list nothing removed")
            return
        del self.transformation[self.names[name]]
        del self.skip[self.names[name]]

    def get_transformation_by_name(self,
                                   name=None):
        if name is None:
            print("WARNING: Must have a name nothing removed")
            return
        if name not in self.names.keys():
            print("WARNING: Name not in list nothing removed")
            return
        return (self.transformation[self.names[name]])

    def execute_transformations(self):
        transform_list = list(self.transformation)
        transform_list.sort()

        for transform_i in transform_list:
            if not self.skip[transform_i]:
                # getattr(data_class, transformations.transformation[transform_i].transformation_name)(
                #     transformations.transformation[transform_i])transform_i
                getattr(self, self.transformation[transform_i].transformation_name)(self.transformation[transform_i])
