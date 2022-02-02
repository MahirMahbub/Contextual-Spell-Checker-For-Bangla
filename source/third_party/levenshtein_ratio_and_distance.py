from typing import Tuple

import numpy as np
from numpy import ndarray


class Levenshtein(object):

    @staticmethod
    def get_levenshtein_ratio_and_distance(s, t) -> Tuple[float, int]:
        """ levenshtein_ratio_and_distance:
            Calculates levenshtein distance between two strings.
            If ratio_calc = True, the function computes the
            levenshtein distance ratio of similarity between two strings
            For all i and j, distance[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        # Initialize matrix of zeros
        rows: int = len(s) + 1
        cols: int = len(t) + 1
        distance: ndarray = np.zeros((rows, cols), dtype=int)
        cost: int = 0
        # Populate matrix of zeros with the indices of each character of both strings
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
        row = None
        col = None
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                    # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
                else:
                    # In order to align the results with those of the Python Levenshtein package, if we choose to
                    # calculate the ratio the cost of a substitution is 2. If we calculate just distance, then the cost
                    # of a substitution is 1.
                    cost = 2
                distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                         distance[row][col - 1] + 1,  # Cost of insertions
                                         distance[row - 1][col - 1] + cost)  # Cost of substitutions
        # Computation of the Levenshtein Distance Ratio
        ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return ratio, distance[row][col]
