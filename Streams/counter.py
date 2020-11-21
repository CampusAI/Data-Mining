import os
import pathlib

from hashlib import md5
import numpy as np
import random
from scipy.integrate import quad
from struct import unpack


class Counter:

    def __init__(self, b):
        self.b = b
        self.p = 2**b
        # self.M = -np.ones(self.p)*np.inf
        self.M = np.zeros(self.p)
        self.bin_repr_len = 32

        # Compute alpha
        def integrand(u):
            return (np.log2((2 + u)/(1 + u)))**self.p
        self.alpha_p = (self.p*quad(integrand, 0, np.inf)[0])**-1

    def add(self, hashed_elem):
        bin_representation = "{0:b}".format(hashed_elem).zfill(self.bin_repr_len)
        i = int(bin_representation[:self.b], 2)  # Position
        p_plus = bin_representation[self.b:].find("1") + 1  # Leading zeros
        p_plus = self.bin_repr_len - self.b + 1 if p_plus == 0 else p_plus
        self.M[i] = max(self.M[i], p_plus)

    def hash_add(self, elem):
        # Hash value
        hashed_elem = unpack("<IIII",md5(elem).digest())[0]
        self.add(hashed_elem)

    def union(self, count):
        """Union of self and another counter (side-effects self.M)
        """
        self.M = np.maximum(self.M, count.M)

    def size(self):
        # m = self.M[self.M > -np.inf]  # Remove - infinity
        Z = np.sum(np.power(2, -self.M))**-1
        return self.alpha_p*Z*(self.p**2)

    def save(self, file):
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        np.save(file + ".npy", self.M)
    
    def load(self, file):
        self.M = np.load(file + ".npy")

    def __eq__(self, count):
        """Whether two counters are equal
        """
        return np.array_equal(self.M, count.M)


if __name__ == "__main__":
    b = 5
    counter = Counter(b=b)

    num_dif_values = 100
    elems = np.array(list(set(
        [random.randint(0, 2**32-1) for _ in range(num_dif_values)])
    ), dtype=np.int32)
    num_dif_values = len(elems)

    iterations = 100000
    for _ in range(iterations):
        elem = np.random.choice(elems)
        counter.hash_add(elem=elem)

    print("Real diff values: ", num_dif_values)
    print("Approx diff values: ", counter.size())