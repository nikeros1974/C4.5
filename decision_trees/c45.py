import math
import logging


class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, path_to_data, path_to_names):
		self.path_to_data = path_to_data
		self.path_to_names = path_to_names
		self.data = []
		self.classes = []
		self.attr_config = {}
		self.attributes = []
		self.tree = None

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
		for index,row in enumerate(self.data):
			for attr_index in range(self.n_attributes()):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def generate_tree(self):
		self.tree = self.__recursive_generate_tree(self.data, self.attributes)

	def __recursive_generate_tree(self, cur_data, cur_attributes):
		allSame = self.allSameClass(cur_data)

		if len(cur_data) == 0:
			#Fail
			return Node(True, "Fail", None)
		elif allSame is not False:
			#return a node with that class
			return Node(True, allSame, None)
		elif len(cur_attributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(cur_data)
			return Node(True, majClass, None)
		else:
			(best,best_threshold,splitted) = self.splitAttribute(cur_data, cur_attributes)
			remainingAttributes = cur_attributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.__recursive_generate_tree(subset, remainingAttributes) for subset in splitted]
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]


	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attr_config[attribute]) == 1 and self.attr_config[attribute][0] == "continuous":
			return False
		else:
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attr_config[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[i] == valuesForAttribute[index]:
							subsets[index].append(row)
							break
				e = gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self,unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def entropy(self, data):
		res = 0
		freq = self.frequency(data)
		if len(data) > 0:
			# entropy
			res = -1 * sum([x*C45.log(x) for x in freq])
			# gini
#			res = sum([x*(1-x) for x in freq])
			# for f in freq:
			# 	res += f*C45.log(f)
			# res *= -1
		return res

	def frequency(self, data):
		num_classes = len(self.classes) * [0, ]
		for row in data:
			class_index = list(self.classes).index(row[-1])
			num_classes[class_index] += 1
		return [x/len(data) for x in num_classes]

	def log(x):
		return 0 if x == 0 else math.log(x,2)


class Node:
	def __init__(self,is_leaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.is_leaf = is_leaf
		self.children = []





