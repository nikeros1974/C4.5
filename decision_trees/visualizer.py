
class Visualizer:

    def __init__(self, indent=''):
        self.indent = indent
        self.attributes = None

    def print(self, trained_tree):
        self.attributes = trained_tree.attributes
        self.__print_tree(trained_tree.tree)

    def __print_tree(self, node=None, indent=None):
        if indent is None:
            indent = self.indent

        if not node.is_leaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    if child.is_leaf:
                        print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + self.attributes[index] + " : ")
                        self.__print_tree(child, indent + "	")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.is_leaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold) + " : ")
                    self.__print_tree(leftChild, indent + "	")

                if rightChild.is_leaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.__print_tree(rightChild, indent + "	")


