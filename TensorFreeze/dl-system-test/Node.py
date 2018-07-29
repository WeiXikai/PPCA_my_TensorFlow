import numpy as np


# This file is for the Node and Operation


class Node(object):
    """Node is a basic element in a tenserflow graph"""

    def __init__(self):
        """
            inputs: the fathers(value source) of the current node
            op: refer to the operation of the current node
            value: value of this node:
                * for constant, the value is only the value :)
                * for placeholder, their value is defined from the feed_dict
                * for variables, their value is stored in a var_list, and when
                run the initialize function, their value will be updated
                * for other calculation node, their value is determined in the run
                process, and update when the graph compute is running
            const_value: if constant exists, this store the constant
            name: for debug
        """
        self.inputs = []
        self.op = None
        self.value = None
        self.const_value = None
        self.name = None

    def __add__(self, other):
        """Add two node will return a new node"""
        if isinstance(other, Node):
            # other is a Node
            new_node = add_op(self, other)
        else:
            # other is a constant
            new_node = add_op_const(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            new_node = sub_op_const(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            assert False, "\033[1;31mTwo nodes don't need __rsub__ :(\033[0m"
        else:
            new_node = rsub_op_const(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_op_const(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = div_op_const(self, other)
        return new_node

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            assert False, "\033[1;31mTwo nodes don't need __rdiv__ :(\033[0m"
        else:
            new_node = rdiv_op_const(self, other)
        return new_node

    # Allow the left-hand-side operation
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        return self.name

    __repr__ = __str__


class Operator(object):
    """Operator is the basic class for the operation of the node"""

    def __call__(self):
        """ create a new node with specific operation """
        new_node = Node()
        new_node.op = self
        return new_node

    # Given the value of the input nodes, compute can give the value of the result
    # virtual function
    def compute(self, node, input_nodes):
        """
        :param node: node that is going to be computed
        :param input_nodes: nodes of input
        :return: the value of the node
        """
        raise NotImplementedError

    # build a gradient tree to calculate the grad of node
    # the current node's grad is given in the output_grad
    # virtual function
    def gradient(self, node, this_grad):
        """
        :param node: the grad of this node will be calculated
        :param this_grad: the current node's grad
        :return: a list: the gradient contribution to the input_node of this node,
                the order is same as the order in the inputs of the node
        """
        raise NotImplementedError


class AddOp(Operator):
    # Operation that add two node into a new node
    def __call__(self, node_left, node_right):
        """
        :param node_left: the left node in the add operation
        :param node_right: the right node in the add operation
        :return: a new node represent node_A + node_B
        """
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s + %s)" % (node_left.name, node_right.name)
        return new_node

    # the implement of the virtual function in the class Operator
    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mAddOp compute args_num is not 2\033[0m"
        node.value = input_nodes[0].value + input_nodes[1].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad, node.inputs[0]), reducereshapesum_op(this_grad, node.inputs[1])]


class AddByConstOp(Operator):
    # Operation that add a node with a constant
    def __call__(self, node_left, const_value):
        """
        :param node_left: the left node of the add operation
        :param const_value: the right const number of the add operation
        :return: a new node represent node_left + const_value
        """
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left]
        new_node.name = "(%s + %s)" % (node_left.name, str(const_value))
        return new_node

    # the implement of the virtual function in the class operator
    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mAddByConstOp compute args_num is not 1\033[0m"
        node.value = input_nodes[0].value + node.const_value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad, node.inputs[0])]


class SubOp(Operator):
    # Op to sub two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s - %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mSubOp compute args_num is not 2\033[0m"
        node.value = input_nodes[0].value - input_nodes[1].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad, node.inputs[0]), reducereshapesum_op(-this_grad, node.inputs[1])]


class SubByConstOp(Operator):
    # Op to sub a node with a constant
    def __call__(self, node_left, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left]
        new_node.name = "(%s - %s)" % (node_left.name, str(const_value))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mSubByConstOp compute args_num is not 2\033[0m"
        node.value = input_nodes[0].value - node.const_value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad, node.inputs[0])]


class RSubByConstOp(Operator):
    # Op to sub a constant with a node
    def __call__(self, node_right, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_right]
        new_node.name = "(%s - %s)" % (str(const_value), node_right.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mRSubByConstOp compute args_num is not 1\033[0m"
        node.value = node.const_value - input_nodes[0].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(-this_grad, node.inputs[0])]


class MulOp(Operator):
    # Op tp multiply two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s * %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mMulOp compute args_num is not 2\033[0m"
        node.value = input_nodes[0].value * input_nodes[1].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(node.inputs[1] * this_grad, node.inputs[0]), node.inputs[0] * this_grad]


class MulByConstOp(Operator):
    # Op to multiply a node with a constant
    def __call__(self, node_left, const_value):
        new_node = Operator.__call__(self)
        new_node.const_value = const_value
        new_node.inputs = [node_left]
        new_node.name = "(%s * %s)" % (node_left.name, str(const_value))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mMulByConstOp compute args_num is not 1\033[0m"
        node.value = input_nodes[0].value * node.const_value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(node.const_value * this_grad, node.inputs[0])]


class DivOp(Operator):
    # Op to div with two nodes
    def __call__(self, node_left, node_right):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left, node_right]
        new_node.name = "(%s / %s)" % (node_left.name, node_right.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mDivOp compute args_num is not 2\033[0m"
        node.value = input_nodes[0].value / input_nodes[1].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(1 / node.inputs[1] * this_grad, node.inputs[0]), reducereshapesum_op(-node.inputs[0] / (node.inputs[1] * node.inputs[1]) * this_grad, node.inputs[1])]


class DivByConstOp(Operator):
    # Op to div a node with a const
    def __call__(self, node_left, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_left]
        new_node.name = "(%s / %s)" % (node_left.name, str(const_value))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mDivByConstOp compute args_num is not 1\033[0m"
        node.value = input_nodes[0].value / node.const_value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad / node.const_value, node.inputs[0])]


class RDivByConstOp(Operator):
    # Op to div a constant with a node
    def __call__(self, node_right, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_right]
        new_node.name = "(%s / %s)" % (node_right.name, str(const_value))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mRDivByConstOp compute args_num is not 1\033[0m"
        node.value = node.const_value / input_nodes[0].value

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(-node.const_value / (node.inputs[0] * node.inputs[0]) * this_grad, node.inputs[0])]


class MatMulOp(Operator):
    """Op to multiply two matrix nodes"""
    def __call__(self, node_left, node_right, trans_left=False, trans_right=False):
        """
        :param node_left: the left matrix
        :param node_right: the right matrix
        :param trans_left: whether to transpose matrix A
        :param trans_right: whether to transpose matrix B
        :return: return a node that refer to the result
        """
        new_node = Operator.__call__(self)
        new_node.left_mat_trans = trans_left
        new_node.right_mat_trans = trans_right
        new_node.inputs = [node_left, node_right]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_left.name, node_right.name, str(trans_left), str(trans_right))
        return new_node

    def compute(self, node, input_nodes):
        if node.left_mat_trans:
            tmp_left_mat = input_nodes[0].value.T
        else:
            tmp_left_mat = input_nodes[0].value
        if node.right_mat_trans:
            tmp_right_mat = input_nodes[1].value.T
        else:
            tmp_right_mat = input_nodes[1].value
        node.value = np.dot(tmp_left_mat, tmp_right_mat)

    def gradient(self, node, this_grad):
        """Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY"""
        return [matmul_op(this_grad, node.inputs[1], False, True), matmul_op(node.inputs[0], this_grad, True, False)]


class ExpOp(Operator):
    """ExpOp is the operator to calculate the exp function"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        # if isinstance(input, Node):
        new_node.inputs = [node_input]
        new_node.name = "exp(%s)" % node_input.name
        # else:
        #     new_node.value = np.exp(input)
        #     new_node.name = "exp(%s)" % (str(input))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mExpOp compute args_num is not 1\033[0m"
        node.value = np.exp(input_nodes[0].value)

    def gradient(self, node, this_grad):
        return [this_grad * np.exp(node.inputs[0])]


class LogOp(Operator):
    """LogOp is the operator to calculate the ln functuon"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "log(%s)" % node_input.name
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mLogOp compute args_num is not 1\033[0m"
        node.value = np.log(input_nodes[0].value)

    def gradient(self, node, this_grad):
        return [1 / node.inputs[0] * this_grad]


class BroadCastOP(Operator):
    """BroadCastOp is a node represent the np.broadcast_to"""
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "broadcast %s to the shape of %s" % (node_A, node_B)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mBroadCastOp compute args_num is not 2\033[0m"
        node.value = np.broadcast_to(node.value, input_nodes[1].shape)

    def gradient(self, node, this_grad):
        return [reducereshapesum_op(this_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class ReduceSumOp(Operator):
    """ReduceSumOp is for the reduce_sum function"""
    def __call__(self, node_input, axis = None, keepdims = None, name = None, reduction_indices = None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.axis = axis
        new_node.name = name
        new_node.reduction_indices = reduction_indices

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mReduceSumOp compute args_num is not 1\033[0m"


class ReduceReshapeSumOp(Operator):
    """ReduceReshapeSum is to reshape node_A to node_B by reduce_sum method"""
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "reduce_reshape_sum %s as shape of %s" % (node_A, node_B)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mReduceReshapeSumOp compute args_num is not 2\033[0m"
        value = input_nodes[0].value
        while len(value.shape) > len(input_nodes[1].value.shape):
            value = np.sum(value, axis = 0)
        for dim in range(len(value.shape)):
            if value.shape[dim] > input_nodes[1].value.shape[dim]:
                value = np.sum(value, axis = dim, keepdims = True)
        node.value = value

    def gradient(self, node, this_grad):
        return [broadcast_op(this_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class ReduceReshapeMeanOp(Operator):
    """ReduceReshapeMeanOp is to reshape node_A to node_B by reduce_mean method """
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "reduce_reshape_mean %s as shape of %s"
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mReduceReshapeMeanOp compute args_num is not 2\033[0m"
        value = input_nodes[0].value
        while len(value.shape) > len(input_nodes[1].value.shape):
            value = np.mean(value, axis = 0)
        for dim in range(len(value.shape)):
            if value.shape[dim] > input_nodes[1].value.shape[dim]:
                value = np.mean(value, axis = dim, keepdims = True)
        node.value = value

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mReduceReshapeMeanOp can't gradient \033[0m"


class PlaceHolderOp(Operator):
    """PlaceHolderOp is to give a input node a position, the value is need feed"""
    def __call__(self, dtype, shape=None, name=None):
        new_node = Operator.__call__(self)
        new_node.shape = shape
        new_node.name = name
        return new_node

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mPlaceholder doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mPlaceholder doesn't support gradient\033[0m"


class VariableOp(Operator):
    """VariableOp is to define a variable, may with a value"""
    def __init__(self):
        # variable_map is the map from variable to its type and value, usually for the initialize function
        # map value is a list, 0 is the type and 1 is the value of this variable
        self.variable_map = {}

    def __call__(self, initial_value=None, name=None, dtype=None):
        new_node = Operator.__call__(self)
        new_node.name = name
        self.variable_map[new_node] = [dtype, initial_value]
        return new_node

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mVariable doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mVariable doesn't support gradient\033[0m"

    def init_variabels(self):
        for key in self.variable_map:
            value = self.variable_map[key]
            if not value[1] is None:
                key.value = value[1] if value[0] is None else value[0](value[1])


class ZerosLikeOp(Operator):
    """ZerosLikeOp is to get a new matrix with the same shape while its elements are all zeros"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "ZerosLike(%s)" % node_input.name
        return new_node

    def compute(self, node, input_nodes):
        """Return zeros_like of the same shape as input"""
        assert isinstance(input_nodes[0].value, np.ndarray)
        node.value = np.zeros(input_nodes[0].value.shape)

    def gradient(self, node, this_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Operator):
    """ZerosLikeOp is to get a new matrix with the same shape while its elements are all ones"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "OnesLike(%s)" % node_input.name
        return new_node

    def compute(self, node, input_nodes):
        assert isinstance(input_nodes[0].value, np.ndarray)
        node.value = np.ones(input_nodes[0].value.shape)

    def gradient(self, node, this_grad):
        return [zeroslike_op(node.inputs[0])]


class GlobalInitializerOp(Operator):
    """for the global_initializer function"""
    def __call__(self):
        new_node = Operator.__call__(self)
        new_node.name = "GlobalInitializer"
        return new_node

    def compute(self, node, input_nodes):
        variable_op.init_variabels()

    def gradient(self, node, this_grad):
        assert False, "GlobalInitializer shouldn't appear in gradint"


# define some object of the class
add_op = AddOp()
add_op_const = AddOp()
sub_op = SubOp()
sub_op_const = SubByConstOp()
rsub_op_const = RSubByConstOp()
mul_op = MulOp()
mul_op_const = MulByConstOp()
div_op = DivOp()
div_op_const = DivByConstOp()
rdiv_op_const = RDivByConstOp()
matmul_op = MatMulOp()
broadcast_op = BroadCastOP()
reducereshapesum_op = ReduceReshapeSumOp()
reducereshapemean_op = ReduceReshapeMeanOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
variable_op = VariableOp()
placeholder_op = PlaceHolderOp()
global_initializer_op = GlobalInitializerOp()
