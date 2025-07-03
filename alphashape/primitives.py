import numpy as np
import warnings
from typing import Callable, Iterator
from scipy.spatial import Delaunay

__all__ = ["circumcenter", "circumradius", "alphasimplices", "critical_alphas"]

PointSet = list[tuple[float, ...]] | np.ndarray

# list of indices of points forming the simplex
Simplex = list[int] | np.ndarray

type AlphaObjective = Callable[[PointSet, float], float]


def circumcenter(points: PointSet) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
    """
    points = np.asarray(points)
    num_rows, _ = points.shape
    A = np.bmat(
        [
            [2 * np.dot(points, points.T), np.ones((num_rows, 1))],
            [np.ones((1, num_rows)), np.zeros((1, 1))],
        ]
    )
    b = np.hstack((np.sum(points * points, axis=1), np.ones((1))))
    return np.linalg.solve(A, b)[:-1]


def circumradius(points: PointSet) -> float:
    """
    Calculate the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    points = np.asarray(points)
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))


def alphasimplices(
    points: PointSet,
) -> Iterator[tuple[Simplex, float]]:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
    """
    coords = np.asarray(points)
    tri = Delaunay(coords)

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix. Likely caused by all points lying in an N-1 space."
            )


def critical_alphas(points: PointSet) -> np.ndarray:
    points = np.asarray(points)
    alphas = set()
    for _, radius in alphasimplices(points):
        if radius > 0:
            alphas.add(1 / radius)
    alphas = np.array(list(alphas))
    alphas.sort()
    return alphas
