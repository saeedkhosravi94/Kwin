import math
from OpenGL.GL import *

class Character:
    def __init__(self, x=0.0, z=0.0, size=0.3, speed=4.0, turn_speed=120.0,
                 color=(0.3, 0.75, 0.4)):
        self.x = x
        self.y = size
        self.z = z
        self.size = size
        self.angle = 0.0
        self.speed = speed
        self.turn_speed = turn_speed
        self.color = color

    def forward_vector(self):
        rad = math.radians(self.angle)
        return math.sin(rad), 0.0, -math.cos(rad)

    def vision(self):
        forward_x, _, forward_z = self.forward_vector()
        eye_offset = self.size * 0.9
        eye = (
            self.x + forward_x * eye_offset,
            self.y + self.size * 0.25,
            self.z + forward_z * eye_offset,
        )
        target = (
            eye[0] + forward_x,
            eye[1],
            eye[2] + forward_z,
        )
        return eye, target, (0, 1, 0), 90.0

    def move(self, direction, dt):
        fx, _, fz = self.forward_vector()
        self.x += fx * self.speed * direction * dt
        self.z += fz * self.speed * direction * dt

    def rotate(self, direction, dt):
        self.angle += self.turn_speed * direction * dt

    def step_move(self, direction, distance=1.5):
        fx, _, fz = self.forward_vector()
        self.x += fx * distance * direction
        self.z += fz * distance * direction

    def step_turn(self, direction, degrees=15.0):
        self.angle += degrees * direction

    def _quad(self, color, verts):
        glColor3f(*color)
        for v in verts:
            glVertex3f(*v)

    def draw(self):
        s = self.size
        r, g, b = self.color
        front_color = (min(r + 0.2, 1.0), min(g + 0.2, 1.0), min(b + 0.2, 1.0))

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(-self.angle, 0, 1, 0)
        glBegin(GL_QUADS)
        self._quad(front_color, [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s)])
        self._quad(self.color, [(-s, -s, s), (-s, s, s), (s, s, s), (s, -s, s)])
        self._quad(self.color, [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)])
        self._quad(self.color, [(s, -s, -s), (s, s, -s), (s, s, s), (s, -s, s)])
        self._quad(self.color, [(-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s)])
        self._quad(self.color, [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)])
        glEnd()

        self._draw_face(s)

        glPopMatrix()

    def _draw_face(self, s):
        z = -s - 0.002
        eye_r = s * 0.16
        eye_y = s * 0.25
        eye_x = s * 0.4
        mouth_hw = s * 0.35
        mouth_hh = s * 0.08
        mouth_y = -s * 0.3
        face_color = (0.05, 0.05, 0.05)

        glBegin(GL_QUADS)
        self._quad(face_color, [(-eye_x - eye_r, eye_y - eye_r, z), (-eye_x + eye_r, eye_y - eye_r, z),
                                 (-eye_x + eye_r, eye_y + eye_r, z), (-eye_x - eye_r, eye_y + eye_r, z)])
        self._quad(face_color, [(eye_x - eye_r, eye_y - eye_r, z), (eye_x + eye_r, eye_y - eye_r, z),
                                 (eye_x + eye_r, eye_y + eye_r, z), (eye_x - eye_r, eye_y + eye_r, z)])
        self._quad(face_color, [(-mouth_hw, mouth_y - mouth_hh, z), (mouth_hw, mouth_y - mouth_hh, z),
                                 (mouth_hw, mouth_y + mouth_hh, z), (-mouth_hw, mouth_y + mouth_hh, z)])
        glEnd()
