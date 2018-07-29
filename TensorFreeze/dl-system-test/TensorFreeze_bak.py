import numpy as np

# These are types of varibles from numpy
float32 = np.float32
float64 = np.float64


class Node (object):
    """This class defines the node in the tensorflow computer graph"""
    def __init__(self):
        self.inputs = [] # the father of the current node
        self.op = None # operation of this code
        self.const_atrr = None # if the node is a constant, this is the constant
        self.name = "" # name of the node, mainly for debug

    def __add__(self, other):
        """Adding two node will return a new node """
        if isinstance(other, Node):
            # the other is a node
            new_node = add_op(self, other)
        else:
            # the other is a const
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """for the * """
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node


    # Allow the left-hand-side add and mul
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        #for debug
        return self.name
    __repr__ = __str__


class Op(object):
    """Op is the operations on the nodes"""
    def __call__(self):
        """create a new node with specified"""
        new_node = Node()
        new_node.op = self
        return new_node
    
    # Given the value of the input nodes, compute can give the value of the result
    # virtual function
    def compute(self, node, input_vals):
        """
        :param node: node that will be computed
        :param input_vals: values of the input node
        :return: the value of the node
        """
        raise NotImplementedError

    # build a gradient tree to calculate the grad of node
    # the current node's grad is given in the output_grad
    # virtual function
    def gradient(self, node, output_grad):
        """
        :param node: the grad of this node will be calculated
        :param output_grad: the current node's grad
        :return: a list: the gradient contribution to the input_node of this node,
                the order is same as the order in the inputs of the node
        """
        raise NotImplementedError

class AddOp(Op):
    # The operation to add two nodes into a new node
    def __call__(self, node_A, node_B):
        """
        :param node_A: the left node in the add operation
        :param node_B: the right node in the add operation
        :return: a new node represent node_A + node_B
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    # the achieve of the virtual function in the Op
    def compute(self, node, input_vals):
        assert len(input_vals) == 2, "AddOP input wrong"
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

class AddByConstOp(Op):
    # the operation to add a node with a number
    def __call__(self, node_A, const_val):
        """
        :param node_A: the left node of the add operation
        :param const_val: the right const number of the add operation
        :return: a new node represent node_A + const
        """
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    # the achieve of the virtual function in the Op
    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "AddConstOp input wrong"
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class MulOp(Op):
    # Op to multiply two nodes.
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    #the achieve of the virtual function in the Op
    def compute(self, node, input_vals):
        assert len(input_vals) == 2, "MulOp input wrong"
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]


class MulByConstOp(Op):
    # Op to multiply a nodes with a constant.
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    # achieve of the virtual function of Op
    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "MulConstOp input wrong"
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """
         matmul is to calculate the multiply of two mat
        :param node_A: the left matrix
        :param node_B: the right matrix
        :param trans_A: whether to transpose matrix A
        :param trans_B: whether to transpose matrix B
        :return: Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        if node.matmul_attr_trans_A:
            tmpA = input_vals[0].T
        else:
            tmpA = input_vals[0]
        if node.matmul_attr_trans_B:
            tmpB = input_vals[1].T
        else:
            tmpB = input_vals[1]
        return np.dot(tmpA, tmpB)

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        tmpmatmulop = MatMulOp()
        tmpA = tmpmatmulop(output_grad, node.inputs[1], False, True)
        tmpB = tmpmatmulop(node.inputs[0], output_grad, True, False)
        return [tmpA, tmpB]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholderOp compute: placeholder values nust be provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        assert False, "placeholderOp gradient: placeholder values nust be provided by feed_dict"


class VariableOp(Op):
    """Op that represent a variable, may with a value"""
    def __init__(self):
        self.variable_list = []
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "Variable compute: placeholder values nust be provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        assert False, "Variable gradient: placeholder values nust be provided by feed_dict"


class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
variable_op = VariableOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()


def Variable(value, dtype = float32, shape = None, name = "variable"):
    # define a variable
    variable_node = variable_op()
    variable_node.type = dtype
    variable_node.shape = shape
    variable_node.name = name
    variable_node.value = value
    return variable_node


def placeholder(dtype = float32, shape = None, name = "placeholder"):
    # define a node that to get input from user
    placeholder_node = placeholder_op()
    placeholder_node.type = dtype
    placeholder_node.shape = shape
    placeholder_node.name = name
    return placeholder_node


class Session:
    def run(self, obj_nodes, feed_dict):
        """
        :param obj_node: nodes whose values need to be computed.
        :param feed_dict: list of variable nodes whose values are supplied by user.
        :return: A list of values for nodes in eval_node_list.
        """
        if isinstance(obj_nodes, list):
            eval_node_list = obj_nodes
        else:
            eval_node_list = [obj_nodes]
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.
        topo_order = find_topo_sort(eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceholderOp):
                continue
            tmp_input_vals = [np.array(node_to_val_map[input]) for input in node.inputs]
            node_to_val_map[node] = node.op.compute(node, tmp_input_vals)
        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in eval_node_list]
        if isinstance(obj_nodes, list):
            return node_val_results
        else:
            return node_val_results[0]

##############################
####### Helper Methods #######
##############################

def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
