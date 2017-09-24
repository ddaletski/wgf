import numpy as np
from copy import deepcopy


class PolynomialFeatures:
    def __init__(self):
        pass

    @staticmethod
    def get_features(X, max_power, mix=False, **kwargs):
        """ get polynomial features for data records
        args:
            X: ndarray        - matrix N*M, row is record and column is feature
            max_power: int    - maximum power of polynome
            mix: bool         - use mixes of powers (e.g. x^2 * y^3)

        kwargs:
            names: [str]      - list of feature names used to generate names for polynomial features
            power_pattern: function(feature: str, power: int) -> str       - function to generate name of n-th power of feature
            product_pattern: function(feature1: str, feature2: str) -> str - function to generate name powers mix of 2 features powers

        returns:
            (data: ndarray, names: [str]) - data: records of data with polynomial features
                                            names: names of these features (e.g.: "x * y^2")
        """

        power_pattern = kwargs["power_pattern"] if kwargs.get("power_pattern") else lambda f, p: "%s^%d" % (f, p)
        product_pattern = kwargs["product_pattern"] if kwargs.get("product_pattern") else lambda f1, f2: "%s * %s" % (f1, f2)
        names = kwargs["names"] if kwargs.get("names") else ["x%d" % i for i in range(X.shape[1])]

        if mix:
            powers = PolynomialFeatures.power_combinations(X.shape[1], max_power)
        else:
            powers = PolynomialFeatures.powers(X.shape[1], max_power)

        poly_names = []
        for power in powers:
            name = ""
            count = 0
            for idx, feature in enumerate(names):
                if power[idx] > 1:
                    if count:
                        name = product_pattern(name, power_pattern(feature, power[idx]))
                    else:
                        name = power_pattern(feature, power[idx])
                    count += 1

                elif power[idx] == 1:
                    if count:
                        name = product_pattern(name, feature)
                    else:
                        name = feature
                    count += 1
            poly_names.append(name)

        _X = np.empty((0, powers.shape[0]))

        for row in X:
            rrow = []
            for power in powers:
                rrow.append(np.product(np.power(row, power)))
            _X = np.append(_X, [rrow], axis=0)

        return _X, poly_names


    @staticmethod
    def power_combinations(nfeatures, max_power):
        """ return all features multiplications with power from 1 to max_power
        args:
            nfeatures: int    - number of features for which to generate powers
            max_power: int    - maximum power of features powers combination

        returns:
            powers: [[int]]   - list of all powers combinations for features
                                powers[i] is list of powers for corresponding features (e.g. 1 for x, 0 for y and 3 for z)
                                0 < sum(powers[i]) <= max_power
        """

        result = []
        def combos(max_power, powers, idx):
            if max_power == 0:
                result.append(powers)
            elif idx == len(powers):
                result.append(powers)
            else:
                combos(max_power-1, powers[:idx] + [powers[idx] + 1] + powers[idx+1:], idx)
                combos(max_power, powers, idx+1)

        combos(max_power, [0] * nfeatures, 0)
        result = sorted(result, key=sum)[1:]
        return np.array(result)


    @staticmethod
    def powers(nfeatures, max_power):
        """ return all powers from 1 to max_power for every feature
        args:
            nfeatures: int    - number of features for which to generate powers
            max_power: int    - maximum power of features powers

        returns:
            powers [[int]]    - the same as for power_combinations but without products of features powers
        """

        result = np.empty((0, nfeatures))
        for power in range(1, max_power+1):
            result = np.concatenate((result, np.eye(nfeatures) * power))
        return result


if __name__ == "__main__":
    X = np.array([[x, y] for x, y in zip(range(10), reversed(range(10)))])
    X, names = PolynomialFeatures.get_features(X, 3, mix=1, names=["x", "y"],
                                               power_pattern=lambda f,p: "(%s^%d)" % (f, p),
                                               product_pattern=lambda f1,f2: f1+f2)
    print(names)
    print(X)