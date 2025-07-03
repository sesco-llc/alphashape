from typing import Callable

from alphashape.primitives import PointSet


Objective = Callable[[PointSet, float], float]


def fast_coverage(points: PointSet, alpha: float) -> float: ...


def component_count(points: PointSet, alpha: float) -> float: ...


def area_threshold(points: PointSet, alpha: float) -> float: ...
