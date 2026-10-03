from OpenGL.GL import GL_QUADS, glBegin, glColor3f, glEnd, glVertex3f


class SugarCube:
    def __init__(self, x, z, size=0.16, height=0.33, color=(0.1, 0.9, 0.15)):
        self.x = x
        self.z = z
        self.size = size
        self.height = height
        self.color = color
        

    @property
    def position(self):
        return self.x, self.z

    def draw(self):
        left, right = self.x - self.size, self.x + self.size
        near, far = self.z - self.size, self.z + self.size
        bottom, top = 0.01, self.height
        faces = (
            ((left, bottom, near), (right, bottom, near),
             (right, top, near), (left, top, near)),
            ((left, bottom, far), (left, top, far),
             (right, top, far), (right, bottom, far)),
            ((left, bottom, near), (left, bottom, far),
             (left, top, far), (left, top, near)),
            ((right, bottom, near), (right, top, near),
             (right, top, far), (right, bottom, far)),
            ((left, top, near), (right, top, near),
             (right, top, far), (left, top, far)),
            ((left, bottom, near), (left, bottom, far),
             (right, bottom, far), (right, bottom, near)),
        )

        glColor3f(*self.color)
        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3f(*vertex)
        glEnd()
