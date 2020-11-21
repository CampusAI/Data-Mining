import numpy as np
from scipy.integrate import quad

class Counter:
    def __init__(self, b):
        self.b = b
        self.p = 2**b
        # self.M = -np.ones(self.p)*np.inf
        self.M = -np.zeros(self.p)
        self.bin_repr_len = 8
        
        # Compute alpha
        def integrand(u):
            return (np.log2((2 + u)/(1 + u)))**self.p
        self.alpha_p = (self.p*quad(integrand, 0, np.inf)[0])**-1

    def add(self, hashed_elem):
        bin_representation = "{0:b}".format(hashed_elem).ljust(self.bin_repr_len, '0')
        # a = bin_representation[:self.b]
        # b = bin_representation[self.b:]
        i = int(bin_representation[:self.b], 2)  # Position
        p_plus = bin_representation[self.b:].find("1") + 1  # Leading zeros
        p_plus = self.bin_repr_len - self.b + 1 if p_plus == 0 else p_plus
        self.M[i] = max(self.M[i], p_plus)

    def size(self):
        # m = self.M[self.M > -np.inf]  # Remove - infinity
        m = self.M
        Z = np.sum(np.power(2, -m))**-1
        return self.alpha_p*Z*(self.p**2)

if __name__ == "__main__":
    b = 5
    counter = Counter(b=b)

    num_dif_values = 1000
    elems = list(range(num_dif_values))
    
    iterations = 100000
    for _ in range(iterations):
        element = np.random.choice(elems)
        hashed_elem = hash(element)
        counter.add(hashed_elem=hashed_elem)

    print("Real diff values: ", num_dif_values)
    print("Approx diff values: ", counter.size())