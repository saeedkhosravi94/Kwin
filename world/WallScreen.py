from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluLookAt
from PIL import Image


class WallScreen:
    def __init__(self, resolution=256):
        self.resolution = resolution

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution, resolution, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, resolution, resolution)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def begin_capture(self, eye, target, up=(0, 1, 0), fov=90.0):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.resolution, self.resolution)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(fov, 1.0, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2], target[0], target[1], target[2], up[0], up[1], up[2])

    def end_capture(self, window_size):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, window_size[0], window_size[1])

    def read_image(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
        glBindTexture(GL_TEXTURE_2D, 0)

        image = Image.frombytes("RGB", (self.resolution, self.resolution), data)
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def draw_on_wall(self, room):
        half_w = room.w * 0.4
        bottom = room.h * 0.2
        top = room.h * 0.85
        z = -room.d + 0.02

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(-half_w, bottom, z)
        glTexCoord2f(1, 0); glVertex3f(half_w, bottom, z)
        glTexCoord2f(1, 1); glVertex3f(half_w, top, z)
        glTexCoord2f(0, 1); glVertex3f(-half_w, top, z)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
