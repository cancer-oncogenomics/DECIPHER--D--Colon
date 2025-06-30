#!/usr/bin/env python3
# Author  : SongTianqiang
# Date    : 2021-12-14
# Fucntion: calculate the area ratio of sub polygon take up the main polygon.
# Update  : 2021-12-16 shenyi propose the concave polygon calculate area may be wrong.
#           So add the full polygon area calculate function.


import math
import argparse
import numpy as np
import matplotlib.path as mplPath


class Point(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_list(self):
        return (self.x, self.y)


class Vector(object):

    def __init__(self, p1, p2):
        self.p1 = Point(*p1)
        self.p2 = Point(*p2)

    def move(self, percent):
        x = self.p1.x + (self.p2.x - self.p1.x) * percent
        y = self.p1.y + (self.p2.y - self.p1.y) * percent
        return (x, y)


class NewPolygon(object):

    def generate(self, edge, origin=(1, 1), r=1):
        points = list()
        for i in range(0, edge):
            x = origin[0] + r * math.cos(2 * math.pi * i / edge)
            y = origin[1] + r * math.sin(2 * math.pi * i / edge)
            p = Point(x, y)
            points.append(p)
        self.points = points
        return points

    def get_points(self):
        return [i.to_list() for i in self.points]

    def get_area(self, points=None):
        # https://stackoverflow.com/questions/34326728/how-do-i-calculate-the-area-of-a-non-convex-polygon
        if points is None:
            if hasattr(self, "points"):
                points = self.points
            else:
                return None
        area = 0
        pts = points.copy()
        pts.append(pts[0])
        n = len(pts)
        for i in range(0, n - 1):
            area += -pts[i].y * pts[i + 1].x + pts[i].x * pts[i + 1].y
        area = 0.5 * abs(area)
        return area

    def get_area2(self, points=None):
        if points is None:
            if hasattr(self, "points"):
                points = self.points
            else:
                return None
        areas = list()
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[0], points[i], points[i + 1]
            tri = Triangle(p1=p1, p2=p2, p3=p3)
            area = self._tri_area(p1, p2, p3)
            areas.append(area)
        return sum(areas)

    def _tri_area(self, p1, p2, p3):
        area = (1 / 2) * abs((p2.x - p1.x) * (p3.y - p1.y) -
                             (p3.x - p1.x) * (p2.y - p1.y))
        return area

    def _get_sub_by_center(self, percents, center):
        if len(percents) == 1:
            percents = [percents[0] for i in range(0, self.edge_nums)]
        else:
            if len(percents) != self.edge_nums:
                raise ValueError("The percents number is error!")
        center = self._point_type(center, "l")
        if self.point_in(center):
            vectors = [Vector(center, self.points[i].to_list()) for i in range(0, self.edge_nums)]
            points = [vectors[i].move(percents[i]) for i in range(0, self.edge_nums)]
            return points
        else:
            raise ValueError("The center not in!")

    def point_in(self, point):
        point = np.array(point)
        points = np.array([p.to_list() for p in self.points])
        bbPath = mplPath.Path(points)
        return bbPath.contains_point(point)

    @staticmethod
    def _point_in(vertices, point):
        bbPath = mplPath.Path(vertices)
        return bbPath.contains_point(point)

    @staticmethod
    def _point_type(point, ntype="l"):
        if ntype == "l":
            if isinstance(point, Point):
                return point.to_list()
            else:
                return point
        elif ntype == "p":
            if isinstance(point, Point):
                return point
            else:
                return Point(*point)


class Triangle(NewPolygon):

    def __init__(self, p1=None, p2=None, p3=None, ntype="equilateral"):
        if p1 is not None:
            if isinstance(p1, (list, tuple)):
                self.points = (Point(*p1), Point(*p2), Point(*p3))
            else:
                self.points = (p1, p2, p3)
        elif ntype == "equilateral":
            points = self.generate(edge=3, origin=(0, 0), r=1)
            self.points = points

    def get_centroid(self):
        x = sum([i.x for i in self.points]) / 3
        y = sum([i.y for i in self.points]) / 3
        return (x, y)

    def get_sub_by_centroid(self, percents):
        center = self.get_centroid()
        center = Point(*center)
        p_x1 = Vector(center, self.points[0])
        p_x2 = Vector(center, self.points[1])
        p_x3 = Vector(center, self.points[2])
        p1 = p_x1.move(percents[0])
        p2 = p_x2.move(percents[1])
        p3 = p_x3.move(percents[2])
        subtri = Triangle(p1=p1, p2=p2, p3=p3)
        return subtri


class Polygon(NewPolygon):

    def __init__(self, points=None, num=3, ntype="equilateral"):
        if points is None:
            points = self.generate(edge=num, origin=(0, 0))
            self.points = points
        else:
            self.points = [self._point_type(p, "p") for p in points]
        if len(self.points) < 3:
            raise ValueError("points number is error!")
        self.edge_nums = len(self.points)

    def get_centroid(self):
        pts = self.points.copy()
        pts.append(pts[0])
        # signed area
        signed = (1 / 2) * sum([pts[i].x * pts[i + 1].y -
                                pts[i + 1].x * pts[i].y
                                for i in range(0, self.edge_nums)])
        Gx = (1 / (6 * signed)) * sum([(pts[i].x + pts[i + 1].x) *
                                       (pts[i].x * pts[i + 1].y -
                                        pts[i + 1].x * pts[i].y)
                                       for i in range(0, self.edge_nums)])
        Gy = (1 / (6 * signed)) * sum([(pts[i].y + pts[i + 1].y) *
                                       (pts[i].x * pts[i + 1].y -
                                        pts[i + 1].x * pts[i].y)
                                       for i in range(0, self.edge_nums)])
        return (Gx, Gy)

    def get_sub_by_edge(self, percents):
        if len(percents) == 1:
            percents = [percents[0] for i in range(0, self.edge_nums)]
        else:
            if len(percents) != self.edge_nums:
                raise ValueError("The percents number is error!")
        points = [p.to_list() for p in self.points]
        points.append(points[0])
        pts = list()
        for i in range(0, self.edge_nums):
            p1 = points[i]
            p2 = points[i + 1]
            percent = percents[i]
            v1 = Vector(p1, p2)
            p_new = v1.move(percent)
            pts.append(p_new)
        return pts

    def get_sub_by_center(self, percents, center=None):
        if len(percents) == 1:
            percents = [percents[0] for i in range(0, self.edge_nums)]
        else:
            if len(percents) != self.edge_nums:
                raise ValueError("The percents number is error!")
        centroid = center
        if centroid is None:
            centroid = self.get_centroid()
        pts = self._get_sub_by_center(percents=percents, center=centroid)
        return pts


def data_check(l):
    if max(l) > 1 or min(l) < 0:
        raise ValueError("Please enter the value between [0, 1]!")


def check_weight(l):
    if min(l) < 0:
        raise ValueError("The weight can't be negative number!")


def combine_metrics(l, weights=None):
    """
    Receive a metrics list, and
    >>> combine_metrics(l = [0.5, 0.6, 0.1, 0.9], weights = [1, 1, 1, 0.5])
    """
    data_check(l)
    poly = Polygon(num=len(l))
    centroid_point = poly.get_centroid()
    if weights is None or isinstance(weights, (int, float)):
        poly = poly
    else:
        check_weight(weights)
        if len(l) != len(weights):
            raise ValueError("The weights length not equal list")
        poly_w_pts = poly.get_sub_by_center(weights)
        poly_w = Polygon(poly_w_pts)
        poly = poly_w
    area_raw = poly.get_area()
    new_points = poly.get_sub_by_center(percents=l, center=centroid_point)
    ploy_sub = Polygon(new_points)
    area = ploy_sub.get_area()
    ratio = area / area_raw
    return ratio