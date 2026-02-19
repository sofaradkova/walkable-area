# synthetic_generator.py
"""
Synthetic 2D polygon & point sampling utilities for pipeline unit tests.

Dependencies:
    pip install shapely numpy pillow

APIs:
- make_rectangle(width, height, origin=(0,0))
- make_realistic_room(width, height, origin=(0,0), seed=None, furniture_density=0.15, ...)
- sample_floor_points(polygon, n_points, seed=None)
- polygon_to_occupancy_image(polygon, resolution=0.05, margin=0.5)

All coordinates are in meters, XY plane. Returned points are Nx3 (x,y,z).
"""
from typing import Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from PIL import Image, ImageDraw

def make_rectangle(width: float, height: float, origin: Tuple[float,float]=(0.0,0.0)) -> Polygon:
    ox, oy = origin
    coords = [(ox, oy), (ox+width, oy), (ox+width, oy+height), (ox, oy+height)]
    return Polygon(coords)

def make_realistic_room(
    width: float,
    height: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    seed: Optional[int] = None,
    furniture_density: float = 0.15,
    add_alcoves: bool = True,
    add_wall_indentations: bool = True,
    min_furniture_size: float = 0.5,
    max_furniture_size: float = 2.0,
    wall_irregularity: float = 0.1
) -> Polygon:
    """
    Generate a realistic room polygon with furniture obstacles and irregular features.
    
    Args:
        width: Base room width in meters
        height: Base room height in meters
        origin: Bottom-left corner of the room
        seed: Random seed for reproducibility
        furniture_density: Fraction of room area that should be occupied by furniture (0.0-0.3)
        add_alcoves: Whether to add wall alcoves (outward protrusions like bay windows)
        add_wall_indentations: Whether to add wall indentations (inward protrusions like closets)
        min_furniture_size: Minimum furniture dimension in meters
        max_furniture_size: Maximum furniture dimension in meters
        wall_irregularity: Amount of random variation to add to walls (0.0-0.2)
    
    Returns:
        Polygon with interior holes representing furniture and obstacles
    """
    rng = np.random.default_rng(seed)
    ox, oy = origin
    
    # Start with base rectangular room
    base_room = make_rectangle(width, height, origin=origin)
    
    # Create irregular room shape by modifying walls
    if wall_irregularity > 0:
        # Get exterior coordinates and add small random variations
        coords = list(base_room.exterior.coords[:-1])  # Exclude duplicate last point
        irregular_coords = []
        for i, (x, y) in enumerate(coords):
            # Add small random offset to make walls slightly irregular
            offset_x = rng.uniform(-wall_irregularity, wall_irregularity)
            offset_y = rng.uniform(-wall_irregularity, wall_irregularity)
            # Keep within bounds
            new_x = max(ox, min(ox + width, x + offset_x))
            new_y = max(oy, min(oy + height, y + offset_y))
            irregular_coords.append((new_x, new_y))
        
        try:
            base_room = Polygon(irregular_coords)
            if not base_room.is_valid:
                base_room = make_rectangle(width, height, origin=origin)
        except:
            base_room = make_rectangle(width, height, origin=origin)
    
    # Add alcoves (outward protrusions like bay windows) - these expand the room
    if add_alcoves and rng.random() < 0.6:  # 60% chance of having an alcove
        alcove_wall = rng.integers(0, 4)
        alcove_depth = rng.uniform(0.4, 1.2)
        alcove_width = rng.uniform(1.0, 2.5)
        
        if alcove_wall == 0:  # bottom wall
            alcove_x = rng.uniform(ox + 0.5, ox + width - alcove_width - 0.5)
            alcove = box(alcove_x, oy - alcove_depth, alcove_x + alcove_width, oy)
        elif alcove_wall == 1:  # right wall
            alcove_y = rng.uniform(oy + 0.5, oy + height - alcove_width - 0.5)
            alcove = box(ox + width, alcove_y, ox + width + alcove_depth, alcove_y + alcove_width)
        elif alcove_wall == 2:  # top wall
            alcove_x = rng.uniform(ox + 0.5, ox + width - alcove_width - 0.5)
            alcove = box(alcove_x, oy + height, alcove_x + alcove_width, oy + height + alcove_depth)
        else:  # left wall
            alcove_y = rng.uniform(oy + 0.5, oy + height - alcove_width - 0.5)
            alcove = box(ox - alcove_depth, alcove_y, ox, alcove_y + alcove_width)
        
        base_room = unary_union([base_room, alcove])
        if not isinstance(base_room, Polygon):
            if hasattr(base_room, 'geoms'):
                base_room = max(base_room.geoms, key=lambda p: p.area)
            else:
                base_room = make_rectangle(width, height, origin=origin)
    
    # Add wall indentations (inward protrusions like closets) - these reduce interior space
    # We'll add these as interior obstacles/holes later, or modify the exterior shape
    if add_wall_indentations and rng.random() < 0.5:  # 50% chance
        # Create an L-shape or add a corner cut to simulate a closet/indentation
        indent_wall = rng.integers(0, 4)
        indent_depth = rng.uniform(0.3, 0.8)
        indent_width = rng.uniform(0.8, 1.8)
        
        # Create indentation by modifying the exterior shape
        coords = list(base_room.exterior.coords[:-1])
        if indent_wall == 0 and len(coords) >= 4:  # bottom wall
            indent_start_x = rng.uniform(ox + 0.3, ox + width - indent_width - 0.3)
            # Modify bottom edge to create indentation
            new_coords = []
            for x, y in coords:
                if abs(y - oy) < 0.01 and indent_start_x <= x <= indent_start_x + indent_width:
                    # Push this point inward
                    new_coords.append((x, y + indent_depth))
                else:
                    new_coords.append((x, y))
            try:
                indented_room = Polygon(new_coords)
                if indented_room.is_valid:
                    base_room = indented_room
            except:
                pass  # Keep original if modification fails
        # Similar logic for other walls could be added
    
    # Ensure we have a valid polygon
    if not isinstance(base_room, Polygon) or not base_room.is_valid:
        base_room = make_rectangle(width, height, origin=origin)
    
    # Add furniture obstacles as interior holes
    furniture_holes = []
    room_area = base_room.area
    target_furniture_area = room_area * furniture_density
    current_furniture_area = 0.0
    
    # Common furniture types with typical sizes and placement preferences
    # Format: (name, (w_min, w_max, h_min, h_max), wall_aligned)
    furniture_types = [
        ('table', (0.8, 1.2, 0.6, 1.0), False),      # dining table - center
        ('desk', (1.0, 1.8, 0.5, 0.7), True),        # desk - wall-aligned
        ('cabinet', (0.4, 0.8, 0.5, 1.2), True),     # cabinet/wardrobe - wall-aligned
        ('sofa', (1.5, 2.5, 0.7, 1.0), True),        # sofa - wall-aligned
        ('bed', (1.5, 2.0, 1.8, 2.2), True),         # bed - wall-aligned
        ('shelf', (0.3, 0.6, 0.8, 1.5), True),       # bookshelf - wall-aligned
    ]
    
    max_attempts = 100
    attempts = 0
    
    while current_furniture_area < target_furniture_area * 0.8 and attempts < max_attempts:
        attempts += 1
        
        # Choose furniture type (select index first to avoid numpy array shape issues)
        idx = rng.integers(0, len(furniture_types))
        ftype, (w_min, w_max, h_min, h_max), wall_aligned = furniture_types[idx]
        
        # Random size within furniture type range, but respect global limits
        fw = rng.uniform(max(w_min, min_furniture_size), min(w_max, max_furniture_size))
        fh = rng.uniform(max(h_min, min_furniture_size), min(h_max, max_furniture_size))
        
        minx, miny, maxx, maxy = base_room.bounds
        wall_gap = rng.uniform(0.05, 0.15)  # Small gap from wall (5-15cm)
        center_margin = 0.3  # Margin for center-placed furniture
        
        # Place furniture based on type
        if wall_aligned:
            # Place furniture against a wall
            wall_side = rng.integers(0, 4)  # 0=bottom, 1=right, 2=top, 3=left
            
            if wall_side == 0:  # Bottom wall
                fx = rng.uniform(minx + wall_gap, maxx - fw - wall_gap)
                fy = miny + wall_gap
            elif wall_side == 1:  # Right wall
                fx = maxx - fw - wall_gap
                fy = rng.uniform(miny + wall_gap, maxy - fh - wall_gap)
            elif wall_side == 2:  # Top wall
                fx = rng.uniform(minx + wall_gap, maxx - fw - wall_gap)
                fy = maxy - fh - wall_gap
            else:  # Left wall
                fx = minx + wall_gap
                fy = rng.uniform(miny + wall_gap, maxy - fh - wall_gap)
        else:
            # Center furniture (like tables) - place more towards center but allow some flexibility
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            room_width = maxx - minx
            room_height = maxy - miny
            
            # Place in center area (within 40% of room center)
            fx = rng.uniform(
                center_x - room_width * 0.2 - fw/2,
                center_x + room_width * 0.2 - fw/2
            )
            fy = rng.uniform(
                center_y - room_height * 0.2 - fh/2,
                center_y + room_height * 0.2 - fh/2
            )
            
            # Ensure it stays within bounds with margin
            fx = max(minx + center_margin, min(maxx - fw - center_margin, fx))
            fy = max(miny + center_margin, min(maxy - fh - center_margin, fy))
        
        # Create furniture polygon
        furniture = box(fx, fy, fx + fw, fy + fh)
        
        # Check if furniture fits inside room and doesn't overlap with existing furniture
        if base_room.contains(furniture):
            overlaps = False
            for existing_furniture in furniture_holes:
                if furniture.intersects(existing_furniture):
                    overlaps = True
                    break
            
            if not overlaps:
                furniture_holes.append(furniture)
                current_furniture_area += furniture.area
    
    # Convert furniture to holes in the polygon
    if furniture_holes:
        # Create polygon with holes
        holes_coords = [list(f.exterior.coords) for f in furniture_holes]
        return Polygon(base_room.exterior.coords, holes_coords)
    else:
        return base_room

def sample_floor_points(polygon: Polygon, n_points: int=5000, seed: int=None, z_noise: float=0.01) -> np.ndarray:
    """
    Uniformly sample n_points inside the polygon (2D) and return Nx3 array (x,y,z_noise).
    Deterministic with seed.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    # rejection sampling - fine for simple polygons and moderate n
    attempts = 0
    while len(pts) < n_points and attempts < n_points * 50:
        attempts += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            z = float(rng.normal(0.0, z_noise))
            pts.append((x, y, z))
    if len(pts) < n_points:
        # if polygon is tiny or sampling failed, tile a grid fallback
        xs = np.linspace(minx + 1e-6, maxx - 1e-6, int(np.ceil(np.sqrt(n_points))))
        ys = np.linspace(miny + 1e-6, maxy - 1e-6, int(np.ceil(np.sqrt(n_points))))
        grid = [(x,y) for x in xs for y in ys if polygon.contains(Point(x,y))]
        pts = [(x,y,float(rng.normal(0,z_noise))) for (x,y) in grid[:n_points]]
    return np.array(pts, dtype=float)

def polygon_to_occupancy_image(polygon: Polygon, resolution: float=0.05, margin: float=0.5) -> Tuple[np.ndarray, Tuple[float,float,float,float]]:
    """
    Rasterize polygon to a binary occupancy image.
    Returns:
      - occ: HxW numpy uint8 array with 1==occupied, 0==free
      - bbox: (minx, miny, maxx, maxy) in world coordinates of the raster grid
    resolution in meters per pixel, margin in meters around polygon bounds.
    """
    minx, miny, maxx, maxy = polygon.bounds
    minx -= margin; miny -= margin; maxx += margin; maxy += margin
    width_m = maxx - minx; height_m = maxy - miny
    W = max(3, int(np.ceil(width_m / resolution)))
    H = max(3, int(np.ceil(height_m / resolution)))
    # Image coordinate system: origin top-left; we'll draw polygon in pixel coords
    scale_x = W / width_m
    scale_y = H / height_m
    def world_to_pix(x,y):
        u = int((x - minx) * scale_x)
        v = int((maxy - y) * scale_y)  # flip vertical
        # clip to image extents
        u = max(0, min(W-1, u))
        v = max(0, min(H-1, v))
        return (u,v)

    # create blank image and draw polygon
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    exterior = [world_to_pix(x,y) for x,y in polygon.exterior.coords]
    draw.polygon(exterior, outline=1, fill=1)
    # holes
    for hole in polygon.interiors:
        hole_pts = [world_to_pix(x,y) for x,y in hole.coords]
        draw.polygon(hole_pts, outline=0, fill=0)
    occ = (np.array(img, dtype=np.uint8) > 0).astype(np.uint8)
    return occ, (minx, miny, maxx, maxy)
