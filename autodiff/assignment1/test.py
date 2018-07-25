import random
from cyaron import *

if __name__ == '__main__':
    for i in range(1, 11):
        n = randint(10000, 99000)
        print(n)
        tree = Graph.tree(n)
        io = IO(str(i) + ".in")
        io.input_write(n)
        io.input_write('\n')
        io.input_writeln(tree.to_str(output=Edge.unweighted_edge))
