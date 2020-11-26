import os
import pathlib

from hashlib import md5
import numpy as np
from tqdm import tqdm
import random
from scipy.integrate import quad
from struct import unpack

class Counter:

    DEFAULT_REGISTERS = 5

    def __init__(self, b=5):
        self.b = b
        self.p = 2**b
        self.M = np.zeros(self.p)
        self.bin_repr_len = 32

        # Compute alpha
        def integrand(u):
            return (np.log2((2 + u)/(1 + u)))**self.p
        self.alpha_p = (self.p*quad(integrand, 0, np.inf)[0])**-1

    def add(self, elem):
        bin_representation = "{0:b}".format(hashed_elem).zfill(self.bin_repr_len)
        bucket = int(bin_representation[:self.b], 2)  # Bucket
        p_plus = bin_representation[self.b:].find("1") + 1  # Leading zeros
        p_plus = self.bin_repr_len - self.b + 1 if p_plus == 0 else p_plus # Deal with all 0's case
        self.M[bucket] = max(self.M[bucket], p_plus)
    def hash_add(self, elem):
        # Hash value
        hashed_elem = unpack("<IIII",md5(elem).digest())[0]
        self.add(hashed_elem)

    def union(self, count):
        """Union of self and another counter (side-effects self.M)
        """
        indices = np.where(count.M > self.M)
        if indices[0].shape[0] > 0:
            self.M[indices[0]] = count.M[indices[0]]
        return indices[0].shape[0]

    def size(self):
        # m = self.M[self.M > -np.inf]  # Remove - infinity
        Z = 1. / np.sum(np.power(2, -self.M))
        E = self.alpha_p*Z*(self.p**2)
        E_star = E
        if E <= 5.*self.p/2.:
            v = np.count_nonzero(self.M == 0)
            if v != 0:
                E_star = self.p * np.log(self.p/v)
        elif E > (2**32)/30:
            E_star = - (2**32) * np.log(1 - E/(2**32))
        return E_star


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
    errors = []

    for _ in tqdm(range(100)):
        counter = Counter(b=5)
        num_dif_values = random.randint(10, 1000)
        elems = np.array(list(set(
            [random.randint(0, 2**32-1) for _ in range(num_dif_values)])
        ), dtype=np.int32)
        num_dif_values = len(elems)

        iterations = random.randint(10, 10000)
        for _ in range(iterations):
            elem = np.random.choice(elems)
            counter.add(elem=elem)

        rel_error = (num_dif_values - counter.size())/num_dif_values
        errors.append(rel_error)

    error_mean = np.average(errors)
    error_std = np.std(errors)

    print(errors)

    print("error_mean", error_mean)
    print("error_std", error_std)