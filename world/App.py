import random
from math import cos, dist, radians, sin

import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_UNSIGNED_BYTE,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glBlendFunc,
    glDrawPixels,
    glEnable,
    glEnd,
    glDisable,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glRasterPos2f,
    glPixelZoom,
    glPopMatrix,
    glPushMatrix,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from models.neat import (
    MOVE_BACKWARD,
    MOVE_FORWARD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    NeatTrainer,
    action_from_features,
)
from world.Room import Room
from world.Character import Character
from world.SugerCube import SugarCube
from world.WallScreen import WallScreen


class App:
    EPISODE_STEPS = 120
    TARGET_COUNT = 6
    GOAL_REWARD = 20.0
    TIME_PENALTY = 0.02
    IDLE_PENALTY = 0.04
    MOVEMENT_REWARD_PER_UNIT = 0.5
    WALL_PENALTY = 2.0
    BUTTON_RECT = pygame.Rect(20, 20, 230, 48)

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
        self.targets = []
        self.neat_trainer = NeatTrainer()
        self.evolving = False
        self.stop_after_generation = False
        self.playback_network = self.neat_trainer.load_best_network()
        self.ui_font = pygame.font.Font(None, 28)

        width, height = self.screen.get_size()
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 1)
        self.spawn_targets(random.Random())

    def handle_input(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.running = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.BUTTON_RECT.collidepoint(event.pos)
            ):
                if self.evolving:
                    self.stop_after_generation = not self.stop_after_generation
                else:
                    self.evolving = True
                    self.stop_after_generation = False

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
        for target in self.targets:
            target.draw()

    def capture_vision(self):
        eye, target, up, fov = self.kwin.vision()
        self.wall_screen.begin_capture(eye, target, up, fov)
        self.draw_scene()
        frame = self.wall_screen.read_image()
        self.wall_screen.end_capture(self.screen.get_size())
        return frame

    def render_display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
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
        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        width, height = self.screen.get_size()
        button = self.BUTTON_RECT
        text = self.ui_font.render(
            (
                "Cancel stop"
                if self.evolving and self.stop_after_generation
                else "Stop after generation"
                if self.evolving
                else "Start evolving"
            ),
            True,
            (255, 255, 255),
        )
        status = self.ui_font.render(
            f"Generation: {self.neat_trainer.population.generation}  "
            f"Best fitness: {self.neat_trainer.best_genome.fitness if self.neat_trainer.best_genome else 'n/a'}",
            True,
            (255, 255, 255),
        )

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if self.evolving:
            glColor3f(0.65, 0.12, 0.12)
        else:
            glColor3f(0.1, 0.4, 0.15)
        glBegin(GL_QUADS)
        glVertex3f(button.left, button.top, 0)
        glVertex3f(button.right, button.top, 0)
        glVertex3f(button.right, button.bottom, 0)
        glVertex3f(button.left, button.bottom, 0)
        glEnd()

        glColor3f(1, 1, 1)
        glRasterPos2f(
            button.left + 12,
            button.top + (button.height + text.get_height()) / 2,
        )
        glPixelZoom(1, 1)
        glDrawPixels(
            text.get_width(),
            text.get_height(),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pygame.image.tobytes(text, "RGBA", True),
        )
        glRasterPos2f(
            button.right + 18,
            button.top + (button.height + status.get_height()) / 2,
        )
        glDrawPixels(
            status.get_width(),
            status.get_height(),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pygame.image.tobytes(status, "RGBA", True),
        )

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def render(self):
        self.capture_vision()
        self.render_display()

    def spawn_target(self, rng):
        for _ in range(1000):
            target = (
                rng.uniform(-self.room.w + 0.5, self.room.w - 0.5),
                rng.uniform(-self.room.d + 0.5, self.room.d - 0.5),
            )
            if (
                dist((self.kwin.x, self.kwin.z), target) >= 2.0
                and all(
                    dist(existing.position, target) >= 1.0
                    for existing in self.targets
                )
            ):
                self.targets.append(SugarCube(*target))
                return
        raise RuntimeError("Could not place a green target in the room")

    def spawn_targets(self, rng):
        self.targets = []
        for _ in range(self.TARGET_COUNT):
            self.spawn_target(rng)

    def evaluate_network(self, network):
        rng = random.Random(self.neat_trainer.population.generation)
        self.kwin.x = rng.uniform(-self.room.w + 1.0, self.room.w - 1.0)
        self.kwin.z = rng.uniform(-self.room.d + 1.0, self.room.d - 1.0)
        self.kwin.angle = rng.uniform(0.0, 360.0)
        self.spawn_targets(rng)
        fitness = 0.0

        for step in range(self.EPISODE_STEPS):
            self.handle_input(0.0)

            features = self.kwin.brain(self.capture_vision())
            action = action_from_features(network, features)
            old_position = (self.kwin.x, self.kwin.z)
            nearest_target = min(
                self.targets,
                key=lambda target: dist(old_position, target.position),
            )
            old_target_distance = dist(old_position, nearest_target.position)

            if action == MOVE_FORWARD:
                self.kwin.step_move(1, distance=0.25)
            elif action == MOVE_BACKWARD:
                self.kwin.step_move(-1, distance=0.25)
            elif action == ROTATE_LEFT:
                self.kwin.step_turn(-1)
            elif action == ROTATE_RIGHT:
                self.kwin.step_turn(1)
            else:
                raise ValueError(f"Unexpected NEAT action: {action}")

            min_x, max_x = -self.room.w + self.kwin.size, self.room.w - self.kwin.size
            min_z, max_z = -self.room.d + self.kwin.size, self.room.d - self.kwin.size
            hit_wall = not (
                min_x <= self.kwin.x <= max_x and min_z <= self.kwin.z <= max_z
            )
            self.kwin.x = max(min_x, min(max_x, self.kwin.x))
            self.kwin.z = max(min_z, min(max_z, self.kwin.z))

            movement_distance = dist(old_position, (self.kwin.x, self.kwin.z))
            fitness -= self.TIME_PENALTY
            if movement_distance < 0.01:
                fitness -= self.IDLE_PENALTY
            else:
                fitness += movement_distance * self.MOVEMENT_REWARD_PER_UNIT
            if hit_wall:
                fitness -= self.WALL_PENALTY

            if dist(
                (self.kwin.x, self.kwin.z), nearest_target.position
            ) < 0.5:
                self.targets.remove(nearest_target)
                self.spawn_target(rng)
                fitness += self.GOAL_REWARD
            elif self.targets:
                fitness += 0.2 * (
                    old_target_distance - dist(
                        (self.kwin.x, self.kwin.z), nearest_target.position
                    )
                )

            if not self.targets:
                break
            self.capture_vision()
            if step % 5 == 0:
                self.render_display()

        return fitness

    def run_playback_step(self, dt):
        if self.playback_network is not None:
            features = self.kwin.brain(self.capture_vision())
            action = action_from_features(self.playback_network, features)
            if action == MOVE_FORWARD:
                self.kwin.move(1, dt)
            elif action == MOVE_BACKWARD:
                self.kwin.move(-1, dt)
            elif action == ROTATE_LEFT:
                self.kwin.rotate(-1, dt)
            elif action == ROTATE_RIGHT:
                self.kwin.rotate(1, dt)

            min_x, max_x = -self.room.w + self.kwin.size, self.room.w - self.kwin.size
            min_z, max_z = -self.room.d + self.kwin.size, self.room.d - self.kwin.size
            self.kwin.x = max(min_x, min(max_x, self.kwin.x))
            self.kwin.z = max(min_z, min(max_z, self.kwin.z))

            collected = next(
                (
                    target
                    for target in self.targets
                    if dist(
                        (self.kwin.x, self.kwin.z), target.position
                    ) < 0.5
                ),
                None,
            )
            if collected is not None:
                self.targets.remove(collected)
                self.spawn_target(random.Random())

        self.capture_vision()
        self.render_display()

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0
                self.handle_input(dt)
                if not self.running:
                    break
                if self.evolving:
                    self.neat_trainer.evolve_generation(self.evaluate_network)
                    if self.stop_after_generation:
                        self.evolving = False
                        self.stop_after_generation = False
                        self.neat_trainer.save_checkpoint()
                        self.playback_network = self.neat_trainer.load_best_network()
                else:
                    self.run_playback_step(dt)
            if self.evolving and self.neat_trainer.best_genome is not None:
                self.neat_trainer.save_checkpoint()
        finally:
            pygame.quit()