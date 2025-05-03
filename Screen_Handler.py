from constants import Consts
from graphics import *


class Screen:
    def __init__(self, width=Consts.DIAMETER, height=Consts.DIAMETER):
        height = height or Consts.HEIGHT
        self.win = GraphWin("String Arts", width, height)
        self.win.setCoords(0, 0, width, height)
        self.objects = []
        self.id_counter = 0

    def draw_circle(self, x=0, y=0, radius=None, circle=None, circle_id=None):
        if circle_id is None:
            circle_id = str(self.id_counter)
            self.id_counter += 1
        if circle is not None:
            circ = Circle(Point(circle.x, circle.y), circle.radius)
        else:
            circ = Circle(Point(x, y), radius)
        self.objects.append((circle_id, circ))
        circ.draw(self.win)

    def draw_line(self, x0, y0, x1, y1, line_id=None):
        if line_id is None:
            line_id = str(self.id_counter)
            self.id_counter += 1
        line_objects = Line(Point(x0, y0), Point(x1, y1))
        self.objects.append((id, line_id))
        line_objects.draw(self.win)

    def delete(self, obj_id):
        for i in self.objects:
            if obj_id == i[0]:
                i[1].undraw()

    def wait_mouse(self):
        self.win.getMouse()

    def close(self):
        self.win.close()