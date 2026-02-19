# tests/test_synthetic_generator.py
import numpy as np
from shapely.geometry import Polygon
from src.synthetic_generator import (
    make_rectangle, make_rectangle_with_hole, make_lshape,
    sample_floor_points, polygon_to_occupancy_image
)

def test_rectangle_area_and_sampling():
    rect = make_rectangle(4.0, 3.0, origin=(0.0,0.0))
    pts = sample_floor_points(rect, n_points=1000, seed=42)
    assert pts.shape == (1000, 3)
    # area analytic should be 12
    assert abs(rect.area - 12.0) < 1e-6

def test_rectangle_with_hole_area():
    outer_w, outer_h = 4.0, 3.0
    # hole at (1.0, 0.5) -> (2.0, 1.2)
    hole = (1.0, 0.5, 2.0, 1.2)
    poly = make_rectangle_with_hole(outer_w, outer_h, hole)
    assert abs(poly.area - (outer_w*outer_h - (hole[2]-hole[0])*(hole[3]-hole[1]))) < 1e-6

def test_lshape_area():
    poly = make_lshape(4.0, 3.0, cut_w=1.5, cut_h=1.0)
    # area should equal outer area minus cut area
    assert abs(poly.area - (4.0*3.0 - 1.5*1.0)) < 1e-6

def test_polygon_to_occupancy_image_consistency():
    rect = make_rectangle(4.0, 3.0, origin=(0.0,0.0))
    occ, bbox = polygon_to_occupancy_image(rect, resolution=0.05, margin=0.1)
    # expect non-empty occupancy and bbox covering the rectangle
    assert occ.sum() > 0
    minx, miny, maxx, maxy = bbox
    assert minx < 0.1 and maxx > 3.9  # rectangle from 0..4 in x
    assert miny < 0.1 and maxy > 2.9

if __name__ == "__main__":
    # run quick smoke tests
    test_rectangle_area_and_sampling()
    test_rectangle_with_hole_area()
    test_lshape_area()
    test_polygon_to_occupancy_image_consistency()
    print("Synthetic generator tests passed.")
