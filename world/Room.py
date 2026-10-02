
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class Room:
    def __init__(self, width=10, depth=10, height=4):
        self.w, self.d, self.h = width, depth, height

    def _quad(self, color, verts):
        glColor3f(*color)
        for v in verts:
            glVertex3f(*v)

    def draw(self):
        w, d, h = self.w, self.d, self.h
        glBegin(GL_QUADS)
        self._quad((0.13, 0.13, 0.17), [(-w, 0, -d), (w, 0, -d), (w, 0, d), (-w, 0, d)])
        self._quad((0.09, 0.08, 0.14), [(-w, h, -d), (w, h, -d), (w, h, d), (-w, h, d)])
        self._quad((0.15, 0.15, 0.22), [(-w, 0, -d), (w, 0, -d), (w, h, -d), (-w, h, -d)])
        self._quad((0.15, 0.15, 0.22), [(-w, 0, d), (w, 0, d), (w, h, d), (-w, h, d)])
        self._quad((0.15, 0.15, 0.22), [(-w, 0, -d), (-w, 0, d), (-w, h, d), (-w, h, -d)])
        self._quad((0.15, 0.15, 0.22), [(w, 0, -d), (w, 0, d), (w, h, d), (w, h, -d)])
        glEnd()


