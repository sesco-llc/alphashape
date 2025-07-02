import itertools
import warnings
from typing import Callable

import numpy as np
import shapely
import trimesh
from packaging import version
from shapely.geometry import MultiLineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from trimesh import Trimesh

from alphashape.primitives import PointSet, alphasimplices, critical_alphas

__all__ = ["alphashape", "optimizealpha", "max_containing_alpha"]


def alphashape(points: PointSet, alpha: float | None = None) -> BaseGeometry | Trimesh:
    """
    Compute the alpha shape (concave hull) of a set of points.  If the number
    of points in the input is three or less, the convex hull is returned to the
    user.  For two points, the convex hull collapses to a `LineString`; for one
    point, a `Point`.

    Args:

      points (list or ``shapely.geometry.MultiPoint`` or \
          ``geopandas.GeoDataFrame``): an iterable container of points
      alpha (float): alpha value

    Returns:

      ``shapely.geometry.Polygon`` or ``shapely.geometry.LineString`` or
      ``shapely.geometry.Point`` or ``geopandas.GeoDataFrame``: \
          the resulting geometry
    """
    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(points) < 4 or (alpha is not None and not callable(alpha) and alpha <= 0):
        if not isinstance(points, MultiPoint):
            points = MultiPoint(list(points))
        result = points.convex_hull
        return result

    # Determine alpha parameter if one is not given
    if alpha is None:
        alpha = optimizealpha(points)

    coords = np.array(points)

    # Create a set to hold unique edges of simplices that pass the radius
    # filtering
    edges = set()

    # Create a set to hold unique edges of perimeter simplices.
    # Whenever a simplex is found that passes the radius filter, its edges
    # will be inspected to see if they already exist in the `edges` set.  If an
    # edge does not already exist there, it will be added to both the `edges`
    # set and the `permimeter_edges` set.  If it does already exist there, it
    # will be removed from the `perimeter_edges` set if found there.  This is
    # taking advantage of the property of perimeter edges that each edge can
    # only exist once.
    perimeter_edges = set()

    for point_indices, circumradius in alphasimplices(coords):
        if callable(alpha):
            resolved_alpha = alpha(point_indices, circumradius)
        else:
            resolved_alpha = alpha

        # Radius filter
        if circumradius < 1.0 / resolved_alpha:
            for edge in itertools.combinations(point_indices, r=coords.shape[-1]):
                if all(
                    e not in edges for e in itertools.combinations(edge, r=len(edge))
                ):
                    edges.add(edge)
                    perimeter_edges.add(edge)
                else:
                    perimeter_edges -= set(itertools.combinations(edge, r=len(edge)))

    if coords.shape[-1] > 3:
        return perimeter_edges
    elif coords.shape[-1] == 3:
        import trimesh

        result = trimesh.Trimesh(vertices=coords, faces=list(perimeter_edges))
        trimesh.repair.fix_normals(result)
        return result

    # Create the resulting polygon from the edge points
    m = MultiLineString([coords[np.array(edge)] for edge in perimeter_edges])
    triangles = list(polygonize(m))
    result = unary_union(triangles)

    return result


def optimizealpha(
    points: PointSet,
    objective: Callable[[PointSet, float], float] | None = None,
    lower: float = 0.0,
    upper: float = np.inf,
    silent: bool = False,
    **kwargs,
) -> float:
    """
    Solve for the alpha parameter.

    Attempt to determine the alpha parameter that best wraps the given set of
    points in one polygon without dropping any points.

    Note:  If the solver fails to find a solution, a value of zero will be
    returned, which when used with the alphashape function will safely return a
    convex hull around the points.

    Args:

        points: an iterable container of points
        max_iterations (int): maximum number of iterations while finding the
            solution
        lower: lower limit for optimization
        upper: upper limit for optimization
        silent: silence warnings

    Returns:

        float: The optimized alpha parameter

    """
    if objective is None:
        objective = max_containing_alpha

    alphas = critical_alphas(points)
    alphas = alphas[(alphas >= lower) & (alphas <= upper)]
    if not alphas:
        if not silent:
            warnings.warn("No critical alphas within bounds", stacklevel=2)
        return lower

    best, score = lower, np.inf
    for alpha in alphas:
        try:
            result = objective(points, alpha)
        except Exception as e:
            if not silent:
                warnings.warn(f"Error evaluating alpha {alpha}: {e}", stacklevel=2)
        else:
            if result < score:
                best, score = alpha, result
    return best


def _testalpha(points: PointSet, alpha: float) -> bool:
    """
    Evaluates an alpha parameter.

    This helper function creates an alpha shape with the given points and alpha
    parameter.  It then checks that the produced shape is a Polygon and that it
    intersects all the input points.

    Args:
        points: data points
        alpha: alpha value

    Returns:
        bool: True if the resulting alpha shape is a single polygon that
            intersects all the input data points.
    """
    polygon = alphashape(points, alpha)
    if isinstance(polygon, shapely.geometry.polygon.Polygon):
        if not isinstance(points, MultiPoint):
            result = MultiPoint(list(points)).geoms
        else:
            result = points
        return all(polygon.intersects(point) for point in result)
    elif isinstance(polygon, trimesh.base.Trimesh):
        return len(polygon.faces) > 0 and all(
            trimesh.proximity.signed_distance(polygon, list(points)) >= 0
        )
    else:
        return False


def max_containing_alpha(points: PointSet, alpha: float) -> float:
    return -alpha if _testalpha(points, alpha) else np.inf
