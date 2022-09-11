import pygame
import dataclasses
from fractions import Fraction as Frac
import networkx as nx
import math
import random
import shapely.geometry
import shapely.ops
pygame.init()
SCREEN = pygame.display.set_mode([1000, 1000])




#arithmetic as a complex number
@dataclasses.dataclass(unsafe_hash=True)
class Point():
    x: Frac
    y: Frac
    def __post_init__(self):
        self.x = Frac(self.x)
        self.y = Frac(self.y)

    def __str__(self):
        return f"Pt({self.x}, {self.y})"

    def __add__(self, other):
        if (cls := type(self)) == type(other):
            return cls(self.x + other.x, self.y + other.y)
        return NotImplemented
    def __neg__(self):
        return type(self)(-self.x, -self.y)
    def __sub__(self, other):
        if (cls := type(self)) == type(other):
            return self + (-other)
        return NotImplemented
    def __mul__(self, other):
        if type(self) == type(other):
            return type(self)(self.x * other.x - self.y * other.y, self.x * other.y + self.y * other.x)
        elif type(other) == Frac:
            return type(self)(self.x * other, self.y * other)
        return NotImplemented
    def __rmul__(self, other):
        if type(other) == Frac:
            return type(self)(other * self.x, other * self.y)
        return NotImplemented
    def recip(self):
        d = self.x ** 2 + self.y ** 2
        if d == 0:
            raise ZeroDivisionError()
        return type(self)(self.x / d, -self.y / d)
        



class NoIntersectError(Exception):
    pass

class InvalidIntersectError(Exception):
    pass


def intersect_lines(a, b):
    assert isinstance(a, Line)
    assert isinstance(b, Line)

    #return p, t, s OR None
    #None if the line segments dont intersect
    #otherwise:
    #p is _a_ Point of intersection (if the lines overlap on a segment, return the midpoint of the segment of overlap
    #t, s are how far along a and b p is

    #start by changing coordinates so that a is the segment (0, 0) - (1, 0)

    b_rel = a.rel(b)

    if b_rel.p.y > 0 and b_rel.q.y > 0:
        raise NoIntersectError()
    elif b_rel.p.y < 0 and b_rel.q.y < 0:
        raise NoIntersectError()
    elif b_rel.p.y == 0 and b_rel.q.y == 0:
        if (b_rel.p.x < 0 and b_rel.q.x < 0) or (b_rel.p.x > 1 and b_rel.q.x > 1):
            raise NoIntersectError()
        raise InvalidIntersectError("Lines should not overlap in the same affine line when testing for intersection")
    else:
        tb = -b_rel.p.y / (b_rel.q.y - b_rel.p.y) #how far along b
        p = tb * b.q + (1 - tb) * b.p
        ta = tb * b_rel.q.x + (1 - tb) * b_rel.p.x
        if 0 <= ta <= 1:
            assert p == ta * a.q + (1 - ta) * a.p
            return p, ta, tb
        else:
            raise NoIntersectError()
        



@dataclasses.dataclass(unsafe_hash=True)
class Line():
    p: Point
    q: Point
    def __post_init__(self):
        assert isinstance(self.p, Point)
        assert isinstance(self.q, Point)
        assert self.p != self.q
    def __str__(self):
        return f"{self.p} - {self.q}"

    def __add__(self, other):
        if type(other) == Point:
            return type(self)(self.p + other, self.q + other)
        return NotImplemented
    def __radd__(self, other):
        if type(other) == Point:
            return type(self)(other + self.p, other + self.q)
        return NotImplemented
    def __sub__(self, other):
        if type(other) == Point:
            return type(self)(self.p - other, self.q - other)
        return NotImplemented
    def __rsub__(self, other):
        if type(other) == Point:
            return type(self)(other - self.p, other - self.q)
        return NotImplemented
    def __mul__(self, other):
        if type(other) == Point:
            return type(self)(self.p * other, self.q * other)
        return NotImplemented
    def __rmul__(self, other):
        if type(other) == Point:
            return type(self)(other * self.p, other * self.q)
        return NotImplemented

    def rel(self, other):
        #coords of other relative to the affine basis where self is mapped to the line from (0, 0) to (1, 0)
        return (other - self.p) * (self.q - self.p).recip()

    def unrel(self, other):
        #convert relative coord back to actual coords
        return self.p + other * (self.q - self.p)

    def affsp(self):
        #return (m, c) for non-vertial lines with gradient m through y=c
        #return (None, d) for vertical lines through x=d
        #the point is that these gradients can be used as unique keys for each linear affine subspace

        if (d := self.p.x) == self.q.x: #vertical lines
            return (None, d)
        else:
            m = (self.q.y - self.p.y) / (self.q.x - self.p.x)
            #c = y-mx
            c = self.p.y - m * self.p.x
            return (m, c)

    def flip(self):
        return type(self)(self.q, self.p)




@dataclasses.dataclass(unsafe_hash=True)
class Ring():
    lines: list[Line]
    def __post_init__(self):
        self.lines = tuple(self.lines)
        for line in self.lines:
            assert isinstance(line, Line)
        self.n = len(self.lines)
        assert self.n >= 3
        self.points = []
        for i in range(self.n):
            self.points.append(self.lines[i].p)
            assert self.lines[i].q == self.lines[(i + 1) % self.n].p

        #rotate so that each oriented ring is unique
        def ord_pt(pt):
            return (pt.x, pt.y)

        off = min(range(self.n), key = lambda i : ord_pt(self.points[i]))
        #rotate so that min_idx -> 0

        self.lines = tuple(self.lines[(i + off) % self.n] for i in range(self.n))
        self.points = tuple(self.points[(i + off) % self.n] for i in range(self.n))

    def __str__(self):
        return " - ".join(str(pt) for pt in self.points)

    def area(self):
        a = Frac(0, 1)
        for line in self.lines:
            a += (line.p.x * line.q.y - line.p.y * line.q.x) * Frac(1, 2)
        return a

    def contains(self, pt):
        #true if pt lies strictly within self
        bx = max([p.x for p in self.points] + [pt.x]) + Frac(1, 1)
        by = max([p.y for p in self.points] + [pt.y]) + Frac(1, 1)
        while True:
            try:
                ray = Line(pt, Point(bx, by))
                #count how many times ray intersects self
                #even => inside
                #odd => outside

                count = 0
                for line in self.lines:
                    try:
                        p, t, s = intersect_lines(ray, line)
                    except NoIntersectError:
                        pass
                    else:
                        if t == 0:
                            return False
                        if s in {0, 1}:
                            raise InvalidIntersectError() #try again with a different ray
                        count += 1
                return count % 2 == 1
            
            except InvalidIntersectError:
                by += Frac(1, 1)

##    def to_tikz(self, exp = 0.0):
##
##        
##
##         "(" + str(self.x) + ", " + str(self.y) + ")"
##        
##        return " -- ".join(pt.to_tikz() for pt in self.points) + " -- cycle"



@dataclasses.dataclass(unsafe_hash=True)
class Face():
    outer: Ring
    inners: frozenset[Ring]
    def __post_init__(self):
        assert isinstance(self.outer, Ring)
        assert self.outer.area() > 0
        self.inners = frozenset(self.inners)
        for ring in self.inners:
            assert isinstance(ring, Ring)
            assert ring.area() < 0

    def __str__(self):
        return str(self.outer) + " \\ (" + " ; ".join(str(inner) for inner in self.inners) + ")"

    def area(self):
        return self.outer.area + sum(inner.area() for inner in self.inners)

    def contains(self, pt):
        if self.outer.contains(pt):
            if not any(inner.contains(pt) for inner in self.inners):
                return True
        return False


@dataclasses.dataclass(unsafe_hash=True)
class Section():
    faces: frozenset[Face]
    def __post_init__(self):
        self.faces = frozenset(self.faces)
        for face in self.faces:
            assert isinstance(face, Face)

    def __str__(self):
        return " | ".join(str(face) for face in self.faces)

    def area(self):
        return sum(face.area() for face in self.faces)
    
    def contains(self, pt):
        return any(face.contains(pt) for face in self.faces)




def faces_from_rings(rings):
    rings = list(rings)
    for ring in rings:
        assert isinstance(ring, Ring)
    faces = {} #{outer ring (with pos area) : [inner rings (with neg area)]}
    for ring in rings:
        if ring.area() > 0:
            faces[ring] = []
    for ring in rings:
        if ring.area() < 0:
            #find the minimal outer ring which contains the inner ring ring
            min_outer_ring = None
            for outer_ring in faces:
                if outer_ring.contains(ring.points[-1]):
                    if min_outer_ring is None or min_outer_ring.contains(outer_ring.points[-1]):
                        min_outer_ring = outer_ring
            if min_outer_ring is not None:
                faces[min_outer_ring].append(ring)
    outer_face_lookup = {outer : Face(outer, faces[outer]) for outer in faces}
    faces = [outer_face_lookup[outer] for outer in faces]
    return faces
        



class Dcel(): #doubly connected line list for stornig planar subdivisions
    def __init__(self, lines):
        lines = list(lines)
        for line in lines:
            assert isinstance(line, Line)

        #directed lines are associated with the face to their left
        self.lines = list(set(lines + [e.flip() for e in lines]))

        #sort lines around each node
        points = {} #{pt : [lines out of pt in clockwise order]}
        for line in self.lines:
            points[line.p] = points.get(line.p, []) + [line]

        def line_valuation(pt, line):
            p, q = line.p, line.q
            if q == pt:
                p, q = q, p
            assert p == pt
            vec = q - pt
            assert vec.x != 0 or vec.y != 0
            if vec.y == 0 and vec.x > 0:
                return (0, 0)
            elif vec.y < 0:
                return (1, vec.x / vec.y)
            elif vec.y == 0 and vec.x < 0:
                return (2, 0)
            else:
                assert vec.y > 0
                return (3, vec.x / vec.y)
            
        #perform the sort
        points = {pt : sorted(points[pt], key = lambda line : line_valuation(pt, line)) for pt in points}
        self.points = points

        rings = []
        root_lines_to_check = set(self.lines)
        while len(root_lines_to_check) != 0:
            root_line = next(iter(root_lines_to_check))
            ring = self.gen_ring(root_line)
            for line in ring.lines:
                root_lines_to_check.remove(line)
            rings.append(ring)
        self.rings = rings

        self.faces = faces_from_rings(self.rings)

    def gen_ring(self, root_line):
        #generate the ring bordered by root_line
        #the ring is the loop of lines starting at root_line and following round to the left

        lines = [root_line]
        while True:
            next_line = self.points[lines[-1].q][(self.points[lines[-1].q].index(lines[-1].flip()) + 1) % len(self.points[lines[-1].q])]
            if next_line == lines[0]:
                break
            else:
                lines.append(next_line)
        return Ring(lines)

    def face_at_point(self, pt):
        for face in self.faces:
            if face.contains(pt):
                return face
        return None



def pg_hsv_colour(h, s, v):
    colour = pygame.Color(0, 0, 0)
    colour.hsva = (h, s, v)
    return colour


COLOURS = {"grey" : pg_hsv_colour(0, 0, 40),
           "red" : pg_hsv_colour(0, 100, 100),
           "orange" : pg_hsv_colour(30, 100, 100),
           "gold" : pg_hsv_colour(45, 100, 100),
           "yellow" : pg_hsv_colour(60, 100, 100),
           "lime" : pg_hsv_colour(90, 100, 100),
           "green" : pg_hsv_colour(120, 100, 100),
           "emerald" : pg_hsv_colour(150, 100, 100),
           "cyan" : pg_hsv_colour(180, 100, 100),
           "blue" : pg_hsv_colour(210, 100, 100),
           "indigo" : pg_hsv_colour(240, 100, 100),
           "violet" : pg_hsv_colour(270, 100, 100),
           "purple" : pg_hsv_colour(285, 100, 100),
           "pink" : pg_hsv_colour(300, 100, 100),
           "fuchsia" : pg_hsv_colour(330, 100, 100)}



def colour_name(name):
    return "my_" + name



class DrawSection():
    def __init__(self, section):
        assert isinstance(section, Section)
        self.section = section
        self.appearance = "normal"

        self.colour = next(iter(COLOURS.keys()))

    @property
    def tikz_colour(self):
        colour = COLOURS[self.colour]
        return "{rgb,255: red, " + str(colour.r) + "; green, " + str(colour.g) + "; blue, " + str(colour.b) + "}"

    @property
    def pg_colour(self):
        colour = COLOURS[self.colour]
        return [colour.r, colour.g, colour.b]
        
    def input_colour(self):
        print("Colours:" + ", ".join(COLOURS.keys()))
        while True:
            colour = input("pg colour:")
            if colour in COLOURS:
                self.colour = colour
                return

    def next_colour(self):
        all_colours = list(COLOURS.keys())
        self.colour = all_colours[(all_colours.index(self.colour) + 1) % len(all_colours)]


    def yield_tikz_shapes(self, idx):
        assert self.appearance in {"normal", "border", "highlight", "black"}
        for face in self.section.faces:
            outer_coords = [[float(pt.x), float(pt.y)] for pt in face.outer.points]
            inners_coords = [[[float(pt.x), float(pt.y)] for pt in inner.points] for inner in face.inners]

            border_thickness = 0.085

            fill_geom = shapely.geometry.Polygon(outer_coords, inners_coords)#.buffer(0.2).buffer(-0.4).buffer(0.2)
            if self.appearance in {"normal"}:
                yield TikzShape(fill_geom, colour_name(self.colour), 16/255, (1, idx)) #light fill
            if self.appearance in {"highlight"}:
                yield TikzShape(fill_geom, colour_name(self.colour), 128/255, (2, idx)) #dark fill

            edge_geom = shapely.ops.unary_union([geom.buffer(border_thickness, join_style = 2) for geom in [fill_geom.exterior] + list(fill_geom.interiors)])
            inner_edge_geom = edge_geom.intersection(fill_geom)
            outer_edge_geom = edge_geom.difference(fill_geom)
            middle_edge_geom = shapely.ops.unary_union([geom.buffer(border_thickness / 3, join_style = 2) for geom in [fill_geom.exterior] + list(fill_geom.interiors)])

            if self.appearance in {"normal"}:
                yield TikzShape(inner_edge_geom, colour_name(self.colour), 1, (3, idx)) #inner colour border
            if self.appearance in {"normal", "border", "black"}:
                yield TikzShape(middle_edge_geom, "black", 1, (4, idx)) #mid black border
            if self.appearance in {"border"}:
                yield TikzShape(outer_edge_geom, colour_name(self.colour), 1, (0, idx)) #outer colour border
            

class TikzShape():
    def __init__(self, geom, colour, opacity, layer):
        geom = shapely.geometry.GeometryCollection([geom])
        self.polys = shapely.geometry.GeometryCollection([g for g in geom.geoms if type(g) == shapely.geometry.Polygon]).geoms
        self.colour = colour
        assert type(self.colour) == str
        self.layer = tuple(layer) #(primary, secondary)
        assert len(self.layer) == 2
        self.opacity = opacity
            
    def to_tikz(self):
        string = "\\fill[fill=" + self.colour + ", opacity=" + str(self.opacity) + ", rounded corners = 0.3cm]"
        for poly in self.polys:
            string += "--".join(f"({round(x, 3)}, {round(y, 3)})" for x, y in poly.exterior.coords[:-1]) + "--cycle"
            for inn in poly.interiors:
                string += "--".join(f"({round(x, 3)}, {round(y, 3)})" for x, y in inn.coords[:-1]) + "--cycle"
            string += ";"      
        return string




class Viewer():
    def __init__(self):
        self.start_pos = None
        self.sel_faces = set()
        self.sel_section = None
        self.draw_sections = []
        self.current_draw_section = None
        
        self.lines = [] #scaffolding lines drawn by the user
        self.dcel = Dcel([]) #the induced planar subdivision

        self.center = [0, 0]
        self.zoom = 100

    def save_state(self):
        #things to save:
        #self.lines
        #self.draw_sections


        #basically boils down to saving lists of lists of ... of lists of lists of ints

        #the layout is as follows:
        #state = [lines, [draw_section]]
        #lines = [line]
        #line = [point, point]

        #draw_section = [section, colour, appearance]
        #section = [faces]
        #faces = [face]
        #face = [ring, [ring]]
        #ring = [point]

        #colour = [int, int, int, int]
        #appearance = 0/1/2/3

        #point = [frac, frac]
        #frac = [int, int]

        def frac_to_state(frac):
            assert isinstance(frac, Frac)
            return [frac.numerator, frac.denominator]

        def point_to_state(point):
            assert isinstance(point, Point)
            return [frac_to_state(point.x), frac_to_state(point.y)]

        def ring_to_state(ring):
            assert isinstance(ring, Ring)
            return [point_to_state(point) for point in ring.points]

        def face_to_state(face):
            assert isinstance(face, Face)
            return [ring_to_state(face.outer), [ring_to_state(inner) for inner in face.inners]]

        def section_to_state(section):
            assert isinstance(section, Section)
            return [face_to_state(face) for face in section.faces]

        def draw_section_to_state(draw_section):
            assert isinstance(draw_section, DrawSection)
            return [section_to_state(draw_section.section), draw_section.colour, {"normal" : 0, "border" : 1, "highlight" : 2, "black" : 3}[draw_section.appearance]]

        def line_to_state(line):
            assert isinstance(line, Line)
            return [point_to_state(line.p), point_to_state(line.q)]

        return str([[line_to_state(line) for line in self.lines], [draw_section_to_state(draw_section) for draw_section in self.draw_sections]])

        

    def load_state(self, state):
        def parse_frac(frac):
            assert type(frac) == list and len(frac) == 2
            return Frac(frac[0], frac[1])
        
        def parse_point(point):
            assert type(point) == list and len(point) == 2
            return Point(parse_frac(point[0]), parse_frac(point[1]))

        def parse_ring(ring):
            assert type(ring) == list
            points = [parse_point(point) for point in ring]
            return Ring([Line(points[i], points[(i+1)%len(points)]) for i in range(len(points))])

        def parse_face(face):
            assert type(face) == list and len(face) == 2
            return Face(parse_ring(face[0]), [parse_ring(ring) for ring in face[1]])

        def parse_section(section):
            assert type(section) == list
            return Section([parse_face(face) for face in section])
        
        def parse_draw_section(draw_section):
            assert type(draw_section) == list and len(draw_section) == 3
            ds = DrawSection(parse_section(draw_section[0]))
            ds.appearance = ["normal", "border", "highlight", "black"][draw_section[2]]
            assert len(draw_section[1]) == 3
            for n in draw_section[1]:
                assert type(n) == int and 0 <= n < 256
            ds.colour = str(draw_section[1]) if str(draw_section[1]) in COLOURS else next(iter(COLOURS.keys())) 
            return ds
        
        def parse_line(line):
            assert type(line) == list and len(line) == 2
            return Line(parse_point(line[0]), parse_point(line[1]))

        def parse_state(state):
            assert type(state) == list and len(state) == 2
            lines, draw_sections = state
            return [parse_line(line) for line in lines], [parse_draw_section(draw_section) for draw_section in draw_sections]
        
        import ast
        try:
            state = ast.literal_eval(state)
        except ValueError:
            print("Invalid Input")
        else:
            lines, draw_sections = parse_state(state)
            self.lines = lines
            self.update()
            self.draw_sections = draw_sections

    def to_tikz(self):
        
        string = ""
        string += "%" + self.save_state() + "\n"

        used_colours = set()
        for draw_section in self.draw_sections:
            used_colours.add(draw_section.colour)

        for used_colour in used_colours:
            colour = COLOURS[used_colour]
            colour = [colour.r, colour.g, colour.b]
            string += "\\definecolor{" + f"my_{used_colour}" + "}{rgb}{" + ",".join(str(round(0.9 * colour[i] / 255, 3)) for i in range(3)) + "}\n"

        string += "\\[\\begin{tikzpicture}[baseline=0]\n"
        string += "\\begin{scope}\n"

        tikz_shapes = []
        for idx, draw_section in enumerate(self.draw_sections):
            tikz_shapes.extend(draw_section.yield_tikz_shapes(idx))
        for tikz_shape in sorted(tikz_shapes, key = lambda shape : shape.layer):
            string += tikz_shape.to_tikz()
        
        string += "\\end{scope}\n"
        string += "\\end{tikzpicture}\\]\n"

        return string

##    @property
##    def sel_section():
##        #the single face given by the union of self.sel_faces

    def from_pg(self, pos):
        pos = [pos[0], SCREEN.get_height() - pos[1]]
        pos = [self.center[i] + (pos[i] - SCREEN.get_size()[i] / 2) / self.zoom for i in [0, 1]]
        return pos
        
    def to_pg(self, pos):
        pos = [(float(pos[i]) - self.center[i]) * self.zoom + SCREEN.get_size()[i] / 2 for i in [0, 1]]
        pos = [pos[0], SCREEN.get_height() - pos[1]]
        return pos

    def update(self):
        #1) combine overlapping lines in the same affine span
        #2) compute all intersection points between pairs of line segments
        #3) split lines at their intersection points to form a graph
        #4) remove all edges whose deletion makes a disconnected graph
        #5) sort lines out of each node in an anticlockwise order
        #6) compute boundaries
        #7) assign outer boundaries as holes for an inner boundary which contains them

        #clear temporary info
        self.sel_faces = set()
        self.sel_section = None
        self.draw_sections = []
        self.current_draw_section = None

        

        affine_spaces = {} #group lines by their affine spaces
        for line in self.lines:
            affsp = line.affsp()
            affine_spaces[affsp] = affine_spaces.get(affsp, []) + [line]

        dist_lines = []

        def combine_lines(affsp, lines):
            #lines is a list of lines all within the linear space affsp
            #at input, the lines in lines may overlap
            #we want to return the simplest collection of line segments covering the same points

            #std line is a choice line within affsp which we will map onto (0, 0)-(1, 0) so that all lines in lines map onto the x-axis

            if affsp[0] is None:
                d = affsp[1]
                #vertical line x=d
                std_line = Line(Point(d, 0), Point(d, 1))
            else:
                m, c = affsp
                #line y=mx+c
                std_line = Line(Point(0, c), Point(1, m+c))

            intervals = set()
            for line in lines:
                rel_line = std_line.rel(line)
                assert rel_line.p.y == rel_line.q.y == 0
                px, qx = rel_line.p.x, rel_line.q.x
                assert px != qx
                if px < qx:
                    intervals.add((px, qx))
                else:
                    intervals.add((qx, px))

            #we are now reduced to simplifying a collection of intervals
            prev_end = -math.inf
            dist_intervals = []
            while len(intervals) != 0:
                min_inter = min(intervals, key = lambda inter : inter[0])
                intervals.remove(min_inter)
                if prev_end < min_inter[0]:
                    dist_intervals.append(min_inter)
                else:
                    dist_intervals[-1] = (dist_intervals[-1][0], max(min_inter[1], dist_intervals[-1][1]))
                prev_end = dist_intervals[-1][1]

            dist_lines = []
            for dist_inter in dist_intervals:
                dist_lines.append(std_line.unrel(Line(Point(dist_inter[0], 0), Point(dist_inter[1], 0))))

            return dist_lines
                        
        for affsp, lines in affine_spaces.items():
            dist_lines.extend(combine_lines(affsp, lines))

        split_fracs = {dist_line : set() for dist_line in dist_lines}
        for i in range(len(dist_lines)):
            for j in range(i+1, len(dist_lines)):
                a, b = dist_lines[i], dist_lines[j]
                try:
                    p, t, s = intersect_lines(a, b)
                except NoIntersectError:
                    pass
                else:
                    split_fracs[a].add(t)
                    split_fracs[b].add(s)
                
        for line in split_fracs:
            split_fracs[line].add(Frac(0, 1))
            split_fracs[line].add(Frac(1, 1))

        graph_lines = []
        for line, fracs in split_fracs.items():
            n = len(fracs)
            fracs = sorted(fracs)
            ps = [t * line.q + (1 - t) * line.p for t in fracs]
            for i in range(n-1):
                graph_lines.append(Line(ps[i], ps[i+1]))

        pointpair_line_lookup = {}
        for line in graph_lines:
            pointpair_line_lookup[(line.p, line.q)] = line
            pointpair_line_lookup[(line.q, line.p)] = line

        graph = nx.Graph()
        for line in graph_lines:
            graph.add_node(line.p)
            graph.add_node(line.q)
            graph.add_edge(line.p, line.q)

        for bridge in nx.bridges(graph):
            graph_lines.remove(pointpair_line_lookup[bridge])

        self.dcel = Dcel(graph_lines)

        

    def draw(self):
        for x in range(-7, 8):
            for y in range(-2, 3):
                pygame.draw.circle(SCREEN, [230, 230, 230], self.to_pg([x, y]), 5)

        pygame.draw.line(SCREEN, [230, 230, 230], self.to_pg([-7, 0]), self.to_pg([7, 0]), 3)

        for line in self.lines:
            p, q = line.p, line.q
            p = self.to_pg([p.x, p.y])
            q = self.to_pg([q.x, q.y])
            
            pygame.draw.line(SCREEN, [200, 200, 200], p, q, 5)
            pygame.draw.circle(SCREEN, [200, 200, 200], p, 5)
            pygame.draw.circle(SCREEN, [200, 200, 200], q, 5)


        def draw_face(face, colour, alpha = 128, appearance = "normal"):
            assert appearance in {"normal", "border", "highlight", "black"}
            inners = face.inners

            surface = pygame.Surface(SCREEN.get_size(), flags = pygame.SRCALPHA)
            
            poses = [self.to_pg([pt.x, pt.y]) for pt in face.outer.points]
            for inner in inners:
                poses.append(poses[0])
                inner_poses = [self.to_pg([pt.x, pt.y]) for pt in inner.points]
                poses.extend(inner_poses)
                poses.append(inner_poses[0])
            if appearance == "border":
                 pygame.draw.lines(surface, colour + [alpha], True, poses, 15)
            elif appearance == "normal":
                pygame.draw.polygon(surface, colour + [alpha], poses)
                pygame.draw.lines(surface, [0, 0, 0] + [255], True, poses, 5)
            elif appearance == "black":
                pygame.draw.lines(surface, [0, 0, 0] + [alpha], True, poses, 10)
            elif appearance == "highlight":
                pygame.draw.polygon(surface, colour + [alpha], poses)
            SCREEN.blit(surface, (0, 0))

##        if len(self.dcel.faces) != 0:
##            face = random.choice(list(self.dcel.faces))
##            draw_face(face, [255, 255, 0])

        for face in self.sel_faces:
            draw_face(face, [0, 0, 0])

        if self.current_draw_section is None:
            if len(self.sel_faces) == 0:
                for draw_section in self.draw_sections:
                    for face in draw_section.section.faces:
                        draw_face(face, draw_section.pg_colour, 128, draw_section.appearance)
        else:
            draw_section = self.draw_sections[self.current_draw_section]
            for face in draw_section.section.faces:
                draw_face(face, draw_section.pg_colour, 255, draw_section.appearance)

                
        


    def event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.current_draw_section = None
                self.start_pos = event.pos

            if event.button in {4, 5}:
                dz = 0.85
                if event.button == 4:
                    dz = 1 / dz

                start = self.from_pg(pygame.mouse.get_pos())
                self.zoom *= dz
                end = self.from_pg(pygame.mouse.get_pos())
                self.center = [self.center[i] + start[i] - end[i] for i in [0, 1]]

            if event.button == 3:
                self.current_draw_section = None
                pos = self.from_pg(event.pos)
                pt = Point(Frac(pos[0]), Frac(pos[1]))
                face = self.dcel.face_at_point(pt)
                if not face is None:
                    if face in self.sel_faces:
                        self.sel_faces.remove(face)
                    else:
                        self.sel_faces.add(face)

                #generate the composite face (maybe redo this bit?) just define union of faces

                border_lines = set()
                for face in self.sel_faces:
                    for ring in [face.outer] + list(face.inners):
                        for line in ring.lines:
                            border_lines.add(line)

                for line in set(border_lines):
                    border_lines.discard(line.flip())
                    
                section_dcel = Dcel(border_lines)

                new_rings = set()
                for root_line in border_lines:
                    if root_line in border_lines:
                        new_rings.add(section_dcel.gen_ring(root_line))

                faces = faces_from_rings(new_rings)
                section = Section(faces)
                self.sel_section = section


        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if not self.start_pos is None:
                    a = self.from_pg(event.pos)
                    b = self.from_pg(self.start_pos)

                    a = [round(a[i]) for i in [0, 1]]
                    b = [round(b[i]) for i in [0, 1]]

                    a = Point(*a)
                    b = Point(*b)

                    if a != b:
                        line = Line(a, b)
                        if line in self.lines:
                            self.lines.remove(line)
                        elif line.flip() in self.lines:
                            self.lines.remove(line.flip())
                        else:
                            self.lines.append(line)
                        self.update()
                    
                self.start_pos = None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.sel_section is not None:
                    if len(self.sel_section.faces) != 0:
                        self.draw_sections.append(DrawSection(self.sel_section))
                        self.sel_faces = set()
                        self.sel_section = None

            if event.key == pygame.K_1:
                if self.current_draw_section is None:
                    if len(self.draw_sections) != 0:
                        self.current_draw_section = 0
                else:
                    self.current_draw_section = (self.current_draw_section - 1) % len(self.draw_sections)

            if event.key == pygame.K_2:
                if self.current_draw_section is None:
                    if len(self.draw_sections) != 0:
                        self.current_draw_section = 0
                else:
                    self.current_draw_section = (self.current_draw_section + 1) % len(self.draw_sections)

            if event.key == pygame.K_DELETE:
                if self.current_draw_section is not None:
                    del self.draw_sections[self.current_draw_section]
                    self.current_draw_section = None

            if event.key == pygame.K_c:
                if self.current_draw_section is not None:
                    self.draw_sections[self.current_draw_section].next_colour()

            if event.key == pygame.K_b:
                if self.current_draw_section is not None:
                    self.draw_sections[self.current_draw_section].appearance = {"normal" : "border", "border" : "highlight", "highlight" : "black", "black" : "normal"}[self.draw_sections[self.current_draw_section].appearance]

            if event.key == pygame.K_i:
                if self.current_draw_section is not None:
                    self.draw_sections[self.current_draw_section].input_colour()

            if event.key == pygame.K_x:
                if self.current_draw_section is not None:
                    self.draw_sections[self.current_draw_section].colour = [0, 0, 0]
                    self.draw_sections[self.current_draw_section].tikz_colour = "black"
                    
            if event.key == pygame.K_t:
                print("\n" * 6)
                print(self.to_tikz())

            if event.key == pygame.K_i:
                self.load_state(input("load:"))
            

                        
                    

            






def run():
    viewer = Viewer()
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            viewer.event(event)

        SCREEN.fill([255, 255, 255])
        viewer.draw()
        pygame.display.update()
        clock.tick(30)



if __name__ == "__main__":
    run()



























