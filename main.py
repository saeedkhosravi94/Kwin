from math import cos, radians, sin

import pygame
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    glClear,
    glClearColor,
    glEnable,
    glLoadIdentity,
    glMatrixMode,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from world.Room import Room
from world.Character import Character
from world.WallScreen import WallScreen


class App:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        self.screen = pygame.display.set_mode(
            (1200, 800), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("NEAT")
        self.clock = pygame.time.Clock()
        self.running = True
        self.room = Room()
        self.user = Character()
        self.kwin = Character(x=2.0, z=2.0, color=(0.75, 0.3, 0.4))
        self.wall_screen = WallScreen()

        width, height = self.screen.get_size()
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 1)

    def handle_input(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.user.move(1, dt)
        if keys[pygame.K_s]:
            self.user.move(-1, dt)
        if keys[pygame.K_a]:
            self.user.rotate(-1, dt)
        if keys[pygame.K_d]:
            self.user.rotate(1, dt)

        self.user.x = max(
            -self.room.w + self.user.size,
            min(self.room.w - self.user.size, self.user.x),
        )
        self.user.z = max(
            -self.room.d + self.user.size,
            min(self.room.d - self.user.size, self.user.z),
        )

    def draw_scene(self):
        self.room.draw()
        self.user.draw()
        self.kwin.draw()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        eye, target, up, fov = self.kwin.vision()
        self.wall_screen.begin_capture(eye, target, up, fov)
        self.draw_scene()
        self.wall_screen.end_capture(self.screen.get_size())

        x, y, z = self.user.x, self.user.y, self.user.z
        angle = radians(self.user.angle)
        camera_direction_x = -sin(angle)
        camera_direction_z = cos(angle)
        wall_distances = [8.0]
        # if camera is hit by the wall move it closer to the user
        if camera_direction_x > 0:
            wall_distances.append((self.room.w - x) / camera_direction_x)
        elif camera_direction_x < 0:
            wall_distances.append((-self.room.w - x) / camera_direction_x)
        if camera_direction_z > 0:
            wall_distances.append((self.room.d - z) / camera_direction_z)
        elif camera_direction_z < 0:
            wall_distances.append((-self.room.d - z) / camera_direction_z)

        camera_distance = max(0.0, min(wall_distances) - 0.3)
        camera_x = x + camera_direction_x * camera_distance
        camera_z = z + camera_direction_z * camera_distance
        gluLookAt(camera_x, y + 2.2, camera_z, x, y, z, 0, 1, 0)
        self.draw_scene()
        self.wall_screen.draw_on_wall(self.room)
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0
                self.handle_input(dt)
                self.render()
        finally:
            pygame.quit()


if __name__ == "__main__":
    App().run()
