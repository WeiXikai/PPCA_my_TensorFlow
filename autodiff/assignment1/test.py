import random
from cyaron import *

if __name__ == '__main__':
    for i in range(1, 11):
        n = randint(1, 1000)
        io = IO(str(i) + ".in")
        io.input_writeln(n)
