import math
import logging
import sys


class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, path_to_data, path_to_names, algo='entropy'):
		self.path_to_data = path_to_data
		self.path_to_names = path_to_names
		self.__set_algo(algo)
		self.data = []
		self.classes = []
		self.attr_config = {}
		self.attributes = []
		self.tree = None

	def __set_algo(self, algo_name):
		if algo_name.lower() in ['entropy', 'gini']:
			self.algo = algo_name
		else:
			logging.log(logging.FATAL, "Algorithm '{:s}' does not exist".format(algo_name))
			sys.exit(1)

	'''
	Data are organized as follows:
	* DATA file contains all the features and target as last column
	* NAMES file contains all the possible target values as first line and a description of each field
	
	The head of the NAMES file is:
	Iris-setosa, Iris-versicolor, Iris-virginica
	sepal length : continuous
	'''
	def fetch_data(self):
		# import configuration file
		logging.log(logging.INFO, "Import config file: {:s}".format(self.path_to_names))
		self.__import_config()

		# not sure this is necessary
		# TODO check if this can be refactored away
		self.attributes = list(self.attr_config.keys())

		# import actual data
		logging.log(logging.INFO, "Import data file: {:s}".format(self.path_to_data))
		self.__import_data()

	def __import_config(self):
		with open(self.path_to_names, "r") as file:
			# The first line is made of all the possible classes
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			# The remaining lines are
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attr_config[attribute] = values
			file.close()

	def __import_data(self):
		with open(self.path_to_data, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				# skip empty rows
				if row != [] or row != [""]:
					self.data.append(row)
			file.close()

	def n_attributes(self):
		return len(self.attr_config.keys())

	def pre_process_data(self):
		for index, row in enumerate(self.data):
			for attr_index in range(self.n_attributes()):
				if not self.is_attribute_discrete(self.attributes[attr_index]):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def generate_tree(self):
		self.tree = self.__recursive_generate_tree(self.data, self.attributes)

	def __recursive_generate_tree(self, cur_data, cur_attributes):
		all_same = C45.all_same_class(cur_data)

		if len(cur_data) == 0:
			#Fail
			return Node(True, "Fail", None)
		elif all_same is not False:
			#return a node with that class
			return Node(True, all_same, None)
		elif len(cur_attributes) == 0:
			#return a node with the majority class
			major_class = self.get_majority_class(cur_data)
			return Node(True, major_class, None)
		else:
			(best, best_threshold, split_sample) = self.split_attribute(cur_data, cur_attributes)
			remainingAttributes = cur_attributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.__recursive_generate_tree(subset, remainingAttributes) for subset in split_sample]
			return node

	def get_majority_class(self, cur_data):
		freq = self.frequency(cur_data)
		return self.classes[freq.index(max(freq))]

	@staticmethod
	def all_same_class(d):
		for r in d:
			if r[-1] != d[0][-1]:
				return False
		return d[0][-1]

	def is_attribute_discrete(self, attribute):
		res = False
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		else:
			res = not (len(self.attr_config[attribute]) == 1 and self.attr_config[attribute][0] == "continuous")
		return res

	def split_attribute(self, cur_data, cur_attributes):
		split_sample = []
		max_ent = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None

		logging.log(logging.INFO, "Current attributes are {:s}".format(str(cur_attributes)))
		for attribute in cur_attributes:
			index_of_attribute = self.attributes.index(attribute)

			if self.is_attribute_discrete(attribute):
				logging.log(logging.INFO, "Processing discrete attribute '{:s}'".format(attribute))
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				values_for_attributes = self.attr_config[attribute]
				subsets = [[] for a in values_for_attributes]
				for row in cur_data:
					for index in range(len(values_for_attributes)):
						if row[index_of_attribute] == values_for_attributes[index]:
							subsets[index].append(row)
							break
				e = self.gain(cur_data, subsets)
				if e > max_ent:
					max_ent = e
					split_sample = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				logging.log(logging.INFO, "Processing continuous attribute '{:s}'".format(attribute))
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				#data are sorted according to the specific attribute
				cur_data.sort(key=lambda x: x[index_of_attribute])
				for j in range(0, len(cur_data) - 1):
					if cur_data[j][index_of_attribute] != cur_data[j+1][index_of_attribute]:
						threshold = (cur_data[j][index_of_attribute] + cur_data[j+1][index_of_attribute]) / 2
						less = []
						greater = []
						for row in cur_data:
							if row[index_of_attribute] > threshold:
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(cur_data, [less, greater])
						# if there is an information gain, store the new values
						if e >= max_ent:
							split_sample = [less, greater]
							max_ent = e
							best_attribute = attribute
							best_threshold = threshold
		res = [best_attribute, best_threshold, split_sample]
		return res

	def gain(self, union_set, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(union_set)
		#calculate impurity before split
		impurityBeforeSplit = self.__score(union_set)
		#calculate impurity after split
		weights = [len(x)/S for x in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.__score(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def __score(self, data):
		res = 0
		if len(data) > 0:
			freq = self.frequency(data)
			# entropy
			if self.algo == 'entropy':
				res = -1 * sum([x*C45.log(x) for x in freq])
			elif self.algo == 'gini':
				res = sum([x*(1-x) for x in freq])
			else:
				logging.log(logging.FATAL, "Should never get here!")
		return res

	def frequency(self, data):
		num_classes = len(self.classes) * [0, ]
		for row in data:
			class_index = list(self.classes).index(row[-1])
			num_classes[class_index] += 1
		return [x/len(data) for x in num_classes]

	@staticmethod
	def log(x):
		return 0 if x == 0 else math.log(x,2)


class Node:
	def __init__(self,is_leaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.is_leaf = is_leaf
		self.children = []





