"""Add Transformation Clases"""


class transformation_class:
	def __init__ (self):
		self.transformation = {}
		self.names = {}
		self.skip = {}

	def add_transformation(self,
						  name = None,
						  order = None,
						  skip = False,
						  transformation = None):	 
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
								name = None):
		if name is None:
			print("WARNING: Must have a name nothing removed")
			return
		if name not in self.names.keys():
			print("WARNING: Name not in list nothing removed")
			return
		del self.transformation[self.names[names]]
		del self.skip[self.names[names]]

	def get_transformation_by_name(self, 
								   name = None):
		if name is None:
			print("WARNING: Must have a name nothing removed")
			return
		if name not in self.names.keys():
			print("WARNING: Name not in list nothing removed")
			return
		return(self.transformation[self.names[name]])


def execute_transformations(data_class,transformations):
	transform_list = list(transformations.transformation)
	transform_list.sort()
	for transform_i in transform_list:
		if not transformations.skip[transform_i]:
			getattr(data_class, transformations.transformation[transform_i].transformation_name)(transformations.transformation[transform_i])
