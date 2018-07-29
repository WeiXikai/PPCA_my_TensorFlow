import numpy as np
from Node import *
from assistance import *

# import dtype from numpy
# float = np.float32
float16 = np.float16
float32 = np.float32
float64 = np.float64
float128 = np.float128
# int = np.int
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

# import operation
placeholder = placeholder_op
Variable = variable_op
global_variables_initializer = global_initializer_op


class Session:

    @staticmethod
    def run(obj_nodes, feed_dict = {}):
        """
        :param obj_nodes: nodes whose values need to be computed.
        :param feed_dict: list of variable nodes whose values are supplied by user.
        :return: A list of values for nodes in eval_node_list.
        """
        eval_node_list = obj_nodes if isinstance(obj_nodes, list) else [obj_nodes]
        feed_placeholder(feed_dict)
        topo_order = find_topo_sort(eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceHolderOp) or isinstance(node.op, VariableOp):
                continue
            node.op.compute(node, [input_node for input_node in node.inputs])
        return [node.value for node in eval_node_list] if isinstance(obj_nodes, list) else obj_nodes.value
