from functools import partial
import itertools
import warnings
from typing import Callable, Iterable

import numpy as np
import shapely
import trimesh
from packaging import version
from shapely.geometry import MultiLineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from trimesh import Trimesh

from alphashape.primitives import PointSet, Simplex, alphasimplices, critical_alphas


def alphashape(
    points: PointSet, alpha: float | None = None
) -> BaseGeometry | Trimesh | set[tuple[Simplex, ...]]:
    """
    Compute the alpha shape (concave hull) of a set of points.  If the number
    of points in the input is three or less, the convex hull is returned to the
    user.  For two points, the convex hull collapses to a `LineString`; for one
    point, a `Point`.

    Supports 2D and 3D point sets

    Args:

      points (list or ``shapely.geometry.MultiPoint`` or \
          ``geopandas.GeoDataFrame``): an iterable container of points
      alpha (float): alpha value

    Returns:

      ``shapely.geometry.Polygon`` or ``shapely.geometry.LineString`` or
      ``shapely.geometry.Point`` or ``geopandas.GeoDataFrame``: \
          the resulting geometry
    """
    coords = np.array(points)

    dim = coords.shape[-1]
    if dim not in {2, 3}:
        raise ValueError(f"Unsupported number of dimensions: {dim}. ")

    # If given a triangle for input, or an alpha value of zero or less,
    # return the convex hull.
    if len(coords) < 4 or (alpha is not None and not callable(alpha) and alpha <= 0):
        if not isinstance(points, MultiPoint):
            result = MultiPoint(list(coords))
        else:
            result = coords
        return result.convex_hull

    # Determine alpha parameter if one is not given
    if alpha is None:
        alpha = optimizealpha(points)

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
    perimeter: set[tuple[Simplex, ...]] = set()

    for point_indices, circumradius in alphasimplices(coords):
        resolved = alpha(point_indices, circumradius) if callable(alpha) else alpha

        # Radius filter
        if circumradius < 1.0 / resolved:
            for edge in itertools.combinations(point_indices, r=dim):
                asdf = itertools.combinations(edge, r=len(edge))
                if all(e not in edges for e in asdf):
                    edges.add(edge)
                    perimeter.add(edge)
                else:
                    perimeter -= set(asdf)

    if dim == 2:
        # Create the resulting polygon from the edge points
        m = MultiLineString([coords[np.array(edge)] for edge in perimeter])
        triangles = list(polygonize(m))
        result = unary_union(triangles)
    else:
        result = Trimesh(vertices=coords, faces=list(perimeter))
        trimesh.repair.fix_normals(result)

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
        objective = max_superset

    alphas = critical_alphas(points)
    alphas = alphas[(alphas >= lower) & (alphas <= upper)]
    if alphas.size == 0:
        if not silent:
            warnings.warn("No critical alphas within bounds", stacklevel=2)
        return lower

    values = _safemap(partial(objective, points), alphas)
    amin = _argmin(values)
    if amin is None:
        if not silent:
            warnings.warn("No feasible alpha found", stacklevel=2)
        return lower
    return alphas[amin]


def is_feasible(points: PointSet, alpha: float) -> bool:
    """
    Evaluates an alpha parameter for feasibility.

    Args:
        points: data points
        alpha: alpha value
    Returns:
        bool: True if the resulting alpha shape intersects all input points.
    """
    polygon = alphashape(points, alpha)

    if not isinstance(points, MultiPoint):
        point_geoms = MultiPoint(list(points)).geoms
    else:
        point_geoms = points.geoms

    match polygon:
        case trimesh.base.Trimesh():
            return len(polygon.faces) > 0 and all(
                trimesh.proximity.signed_distance(polygon, list(points)) >= 0
            )
        case set():
            return len(polygon) > 0
        case polygon if hasattr(polygon, "is_empty") and polygon.is_empty:
            return False
        case _:  # 2D Shapely geometry
            return all(polygon.covers(point) for point in point_geoms)


def polygon_contains_or_intersects(polygon: BaseGeometry, point: BaseGeometry) -> bool:
    """
    Check if a polygon contains or touches a point.

    Args:
        polygon: A polygon geometry.
        point: A point geometry.

    Returns:
        bool: True if the polygon contains or intersects the point, False otherwise.
    """
    return polygon.contains(point) or polygon.touches(point)


def max_superset(points: PointSet, alpha: float) -> float:
    """Maximize alpha while containing all points. Expensive but thorough."""
    return alpha if is_feasible(points, alpha) else -np.inf


def max_area(points: PointSet, alpha: float) -> float:
    """Maximize total area. Fast, good for substantial shapes."""
    shape = alphashape(points, alpha)
    try:
        area = shape.area
    except AttributeError:
        # Handle cases where shape does not have an area attribute
        return -np.inf
    return area if area > 0 else -np.inf


def max_compactness(points: PointSet, alpha: float) -> float:
    """Maximize area/perimeter² ratio. Prefers compact shapes."""
    shape = alphashape(points, alpha)
    try:
        area = shape.area
        length = shape.length
    except AttributeError:
        # Handle cases where shape does not have area or length attributes
        return -np.inf
    if area > 0 and length > 0:
        return area / length**2
    return -np.inf


def max_nonempty(points: PointSet, alpha: float) -> float:
    """Simple non-empty check. Fastest option."""
    shape = alphashape(points, alpha)
    if hasattr(shape, "is_empty") and not shape.is_empty:
        return alpha
    elif isinstance(shape, trimesh.base.Trimesh) and len(shape.faces) > 0:
        return alpha
    elif isinstance(shape, set) and len(shape) > 0:
        return alpha
    return -np.inf


def max_unfragmented(points: PointSet, alpha: float) -> float:
    """Penalize many components. Good for avoiding fragmentation."""
    shape = alphashape(points, alpha)
    if hasattr(shape, "is_empty") and shape.is_empty:
        return -np.inf

    num_components = len(shape.geoms) if hasattr(shape, "geoms") else 1
    return alpha - num_components


def max_bbox_area(points: PointSet, alpha: float, threshold: float = 0.1) -> float:
    """Fast area-based selection. Good middle ground."""
    shape = alphashape(points, alpha)
    if hasattr(shape, "area") and shape.area > 0:
        bbox = MultiPoint(points).bounds
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if shape.area > threshold * bbox_area:
            return alpha
    return -np.inf


def _argmin(it: Iterable) -> int:
    """
    Returns the index of the minimum value in an iterable.

    Args:
        it: An iterable containing numerical values.

    Returns:
        int: The index of the minimum value in the iterable.
    """

    def second(tup: tuple):
        return tup[1]

    amin, _ = min(enumerate(it), key=second)
    return amin


def _safemap(f: Callable, it: Iterable) -> Iterable:
    """
    Applies a function to each element in an iterable and returns a list of results.
    """
    for item in it:
        try:
            yield f(item)
        except Exception as e:
            warnings.warn(f"Error applying function to item {item}: {e}", stacklevel=2)
