from assistance import *
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

    def __neg__(self):
        new_node = rsub_op_const(self, 0)
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
        return [reduce_reshape_sum_op(this_grad, node.inputs[0]), reduce_reshape_sum_op(this_grad, node.inputs[1])]


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
        return [reduce_reshape_sum_op(this_grad, node.inputs[0])]


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
        return [reduce_reshape_sum_op(this_grad, node.inputs[0]), reduce_reshape_sum_op(-this_grad, node.inputs[1])]


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
        return [reduce_reshape_sum_op(this_grad, node.inputs[0])]


class RSubByConstOp(Operator):
    # Op to sub a constant with a node
    def __call__(self, node_right, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_right]
        new_node.const_value = const_value
        new_node.name = "(%s - %s)" % (str(const_value), node_right.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mRSubByConstOp compute args_num is not 1\033[0m"
        node.value = node.const_value - input_nodes[0].value

    def gradient(self, node, this_grad):
        return [reduce_reshape_sum_op(-this_grad, node.inputs[0])]


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
        return [reduce_reshape_sum_op(node.inputs[1] * this_grad, node.inputs[0]), reduce_reshape_sum_op(node.inputs[0] * this_grad, node.inputs[1])]


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
        return [reduce_reshape_sum_op(node.const_value * this_grad, node.inputs[0])]


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
        return [reduce_reshape_sum_op(1 / node.inputs[1] * this_grad, node.inputs[0]), reduce_reshape_sum_op(-node.inputs[0] / (node.inputs[1] * node.inputs[1]) * this_grad, node.inputs[1])]


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
        return [reduce_reshape_sum_op(this_grad / node.const_value, node.inputs[0])]


class RDivByConstOp(Operator):
    # Op to div a constant with a node
    def __call__(self, node_right, const_value):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_right]
        new_node.const_value = const_value
        new_node.name = "(%s / %s)" % (node_right.name, str(const_value))
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mRDivByConstOp compute args_num is not 1\033[0m"
        node.value = node.const_value / input_nodes[0].value

    def gradient(self, node, this_grad):
        return [reduce_reshape_sum_op(-node.const_value / (node.inputs[0] * node.inputs[0]) * this_grad, node.inputs[0])]


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
        """Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY

                           if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
                           if Y=A B^T, then dA=dY B, dB=dY^T dA
                           if Y=A^T B^T, then dA=B dY^T, dB=dY^T A
        """

        return [matmul_op(this_grad, node.inputs[1], False, True ^ node.right_mat_trans), matmul_op(node.inputs[0], this_grad, True ^ node.left_mat_trans, False)]


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
        return [this_grad * exp_op(node.inputs[0])]


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
    def __call__(self, node_A, node_B, reduce_axis = None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.reduce_axis = reduce_axis
        new_node.name = "broadcast(%s, %s)" % (node_A, node_B)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mBroadCastOp compute args_num is not 2\033[0m"
        input0_value = input_nodes[0].value
        input1_value = input_nodes[1].value
        input0_shape = np.shape(input0_value)
        if node.reduce_axis is not None:
            tmp_shape = list(input0_shape)
            for dim in node.reduce_axis:
                tmp_shape.insert(dim, 1)
            input0_shape = tuple(tmp_shape)
            input0_value = np.reshape(input0_value, input0_shape)

        # if len(np.shape(input0_value)) < len(np.shape(input1_value)):
        #     flag = True
        #     for dim in range(0, len(np.shape(input0_value))):
        #         if np.shape(input1_value)[dim] != np.shape(input0_value)[dim]:
        #             flag = False
        #             break
        #     tmp_shape = np.shape(input0_value)
        #     if flag:
        #         while (len(tmp_shape)) < len(np.shape(input1_value)):
        #             tmp_shape = tmp_shape + (1, )
        #     input0_value.resize(tmp_shape)
        node.value = np.broadcast_to(input0_value, np.shape(input1_value))

    def gradient(self, node, this_grad):
        return [reduce_reshape_sum_op(this_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class ReduceSumOp(Operator):
    """ReduceSumOp is for the reduce_sum function"""
    def __call__(self, node_input, axis = None, keepdims = np._NoValue, reduction_indices = None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.axis = axis
        new_node.keepdims = keepdims
        new_node.name = "reduce_sum(%s)" % node_input.name
        new_node.reduction_indices = reduction_indices
        if new_node.axis is None and new_node.reduction_indices is not None:
            new_node.axis = tuple(new_node.reduction_indices)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 1, "\033[1;31mReduceSumOp compute args_num is not 1\033[0m"
        node.value = np.sum(input_nodes[0].value, axis = node.axis, keepdims = node.keepdims)

    def gradient(self, node, this_grad):
        return [broadcast_op(this_grad, node.inputs[0], reduce_axis = node.axis)]


class ReduceMeanOp(Operator):
    """ReduceMeanOp is for the reduce_mean function"""
    def __call__(self, node_input, axis=None, keepdims=np._NoValue, reduction_indices=None):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.axis = axis
        new_node.keepdims = keepdims
        new_node.name = "reduce_mean(%s)" % node_input.name
        new_node.reduction_indices = reduction_indices
        if new_node.axis is None and new_node.reduction_indices is not None:
            new_node.axis = tuple(new_node.reduction_indices)
        return new_node

    def compute(self, node, input_nodes):
        node.value = np.mean(input_nodes[0].value, axis = node.axis, keepdims = node.keepdims)

    def gradient(self, node, this_grad):
        return [broadcast_op(this_grad / reduce_sum_op(oneslike_op(node.inputs[0]), keepdims = node.keepdims), node.inputs[0], reduce_axis = node.axis)]


class ReduceReshapeSumOp(Operator):
    """ReduceReshapeSum is to reshape node_A to node_B by reduce_sum method"""
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "reduce_reshape_sum (%s, %s)" % (node_A, node_B)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mReduceReshapeSumOp compute args_num is not 2\033[0m"
        value_A = input_nodes[0].value
        value_B = input_nodes[1].value
        while len(np.shape(value_A)) > len(np.shape(value_B)):
            value_A = np.sum(value_A, axis = 0)
        for dim in range(len(np.shape(value_A))):
            if np.shape(value_A)[dim] > np.shape(value_B)[dim]:
                value_A = np.sum(value_A, axis = dim, keepdims = True)
        node.value = value_A

    def gradient(self, node, this_grad):
        return [broadcast_op(this_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class ReduceReshapeMeanOp(Operator):
    """ReduceReshapeMeanOp is to reshape node_A to node_B by reduce_mean method """
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "reduce_reshape_mean(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_nodes):
        assert len(input_nodes) == 2, "\033[1;31mReduceReshapeMeanOp compute args_num is not 2\033[0m"
        value_A = input_nodes[0].value
        value_B = input_nodes[1].value
        while len(np.shape(value_A)) > len(np.shape(value_B)):
            value_A = np.mean(value_A, axis = 0)
        for dim in range(len(np.shape(value_A))):
            if np.shape(value_A)[dim] > np.shape(value_B)[dim]:
                value_A = np.mean(value_A, axis = dim, keepdims = True)
        node.value = value_A

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mReduceReshapeMeanOp can't gradient \033[0m"


class SoftMaxOp(Operator):
    """SoftMaxOp is for the softmax function"""
    def __call__(self, input_node):
        exp_input = exp_op(input_node)
        return exp_input / reduce_sum_op(exp_input, axis = 1)

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mSoftMaxNode won't compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mSoftMaxNode won't gradient\033[0m"


class PlaceHolderOp(Operator):
    """PlaceHolderOp is to give a input node a position, the value is need feed"""
    def __call__(self, dtype = np.float32, shape=None, name=None):
        new_node = Operator.__call__(self)
        new_node.shape = shape
        new_node.name = name
        new_node.dtype = dtype
        return new_node

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mPlaceholder doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        return None


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
        return None

    def init_variabels(self):
        for key in self.variable_map:
            value = self.variable_map[key]
            if not value[1] is None:
                key.value = value[1] if value[0] is None else value[0](value[1])


class ZerosTensorOp(Operator):
    """ZerosTensorOp is to get a tensor with the specific shape which elements are all zeros"""
    def __call__(self, shape, dtype = np.float32, name = None):
        return np.zeros(shape, dtype = dtype)

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mZerosTensorOp doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mZerosTensorOp doesn't support gradient\033[0m"


class OnesTensorOp(Operator):
    """OnesTensorOp is to get a tensor with the specific shape which elements are all ones"""
    def __call__(self, shape, dtype = np.float32, name = None):
        return np.ones(shape, dtype = dtype)

    def compute(self, node, input_nodes):
        assert False, "\033[1;31mOnesTensorOp doesn't support compute\033[0m"

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mOnesTensorOp doesn't support gradient\033[0m"


class ZerosLikeOp(Operator):
    """ZerosLikeOp is to get a new node of matrix with the same shape while its elements are all zeros"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "ZerosLike(%s)" % node_input.name
        return new_node

    def compute(self, node, input_nodes):
        node.value = np.zeros(np.shape(input_nodes[0].value))

    def gradient(self, node, this_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Operator):
    """OnesLikeOp is to get a new matrix with the same shape while its elements are all ones"""
    def __call__(self, node_input):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_input]
        new_node.name = "OnesLike(%s)" % node_input.name
        return new_node

    def compute(self, node, input_nodes):
        node.value = np.ones(np.shape(input_nodes[0].value))

    def gradient(self, node, this_grad):
        return [zeroslike_op(node.inputs[0])]


class AssignOp(Operator):
    """for the tf.assign operator"""
    def __call__(self, node_assign, obj_value):
        new_node = Operator.__call__(self)
        if isinstance(obj_value, Node):
            new_node.inputs = [node_assign, obj_value]
        else:
            new_node.inputs = [node_assign]
        new_node.name = "Assign node %s" % node_assign.name
        new_node.assign_value = obj_value
        return new_node

    def compute(self, node, input_nodes):
        if len(input_nodes) == 2:
            input_nodes[0].value = input_nodes[1].value
        else:
            input_nodes[0].value = np.array(node.assign_value) if isinstance(node.assign_value, list) else node.assign_value

    def gradient(self, node, this_grad):
        return None


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

class ArgMaxOp(Operator):
    """ArgMaxOp is for the argmax function"""
    def __call__(self, input_node, axis = None, name = None, dimension = None, output_type = np.int64):
        new_node = Operator.__call__(self)
        new_node.inputs = [input_node]
        new_node.axis = axis
        new_node.dimension = dimension
        new_node.output_type = output_type
        if name is not None:
            new_node.name = name
        else:
            new_node.name = "Argmax(%s)" % input_node.name
        return new_node

    def compute(self, node, input_nodes):
        node.value = node.output_type(np.argmax(input_nodes[0].value, axis = node.axis))

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mArgmaxOp shouldn't appear in gradient\033[0m"


class EqualOp(Operator):
    """EqualOp is for the equal function"""
    def __call__(self, node_A, node_B):
        new_node = Operator.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "equal(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_nodes):
        node.value = input_nodes[0].value == input_nodes[1].value

    def gradient(self, node, this_grad):
        assert False, "\033[1;31mEqualOp shouldn't appear in gradient\033[0m"


class CastOp(Operator):
    """CastOp is for the cast function"""
    def __call__(self, cast_input, dtype):
        new_node = Operator.__call__(self)
        new_node.inputs = [cast_input]
        new_node.dtype = dtype
        if isinstance(cast_input, Node):
            new_node.name = "cast(%s)" % cast_input.name
        else:
            new_node.name = "cast"
        return new_node

    def compute(self, node, input_nodes):
        if isinstance(input_nodes[0], Node):
            node.value = node.dtype(input_nodes[0].value)
        else:
            node.value = node.dtype(input_nodes[0])

    def gradient(self, node, this_grad):
        return [this_grad]


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
exp_op = ExpOp()
log_op = LogOp()
broadcast_op = BroadCastOP()
reduce_sum_op = ReduceSumOp()
reduce_mean_op = ReduceMeanOp()
reduce_reshape_sum_op = ReduceReshapeSumOp()
reduce_reshape_mean_op = ReduceReshapeMeanOp()
softmax_op = SoftMaxOp()
zerostensor_op = ZerosTensorOp()
onestensor_op = OnesTensorOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
variable_op = VariableOp()
placeholder_op = PlaceHolderOp()
assign_op = AssignOp()
global_initializer_op = GlobalInitializerOp()
argmax_op = ArgMaxOp()
equal_op = EqualOp()
cast_op = CastOp()

