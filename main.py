from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class Type:
    name: str
    color: tuple
    unique: bool = False
    immutable: bool = False
    algorithm: bool = False
    box = None


class TYPES:
    DEFAULT = Type("DEFAULT", (80, 80, 80))
    WALL = Type("WALL", (120, 120, 120), immutable=True)
    START = Type("START", (0, 200, 200), unique=True, immutable=True)
    TARGET = Type("TARGET", (200, 200, 0), unique=True, immutable=True)
    QUEUED = Type("QUEUED", (200, 0, 0), algorithm=True)
    VISITED = Type("VISITED", (0, 200, 0), algorithm=True)
    PATH = Type("PATH", (0, 0, 200), algorithm=True)


class Box:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self._type = TYPES.DEFAULT

        self.neighbors = []
        self.prior = None

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_type):
        # Banned operation
        if new_type.unique and self._type is not TYPES.DEFAULT:
            print(f"Cannot allocate {new_type} for {self._type} box")
            return

        if new_type.algorithm and self._type.immutable:
            return

        # Change the unique type
        if self._type.unique:
            self._type.box = None

        # Set a unique type
        if new_type.unique:
            if new_type.box is not None:
                new_type.box._type = TYPES.DEFAULT
            new_type.box = self

        self._type = new_type


class Game:
    def __init__(self, window, columns=20, rows=20):
        self.window = window
        self.columns = columns
        self.rows = rows

        Box.width = window.get_width() // columns
        Box.height = window.get_height() // rows

        # Create Grid
        self.grid = np.array([[Box(i, j) for j in range(rows)] for i in range(columns)])

    def set_neighbors(self):
        for box in self.grid.ravel():
            candidates = []
            if box.i > 0:
                candidates.append(self.grid[box.i - 1][box.j])
            if box.i < self.columns - 1:
                candidates.append(self.grid[box.i + 1][box.j])
            if box.j > 0:
                candidates.append(self.grid[box.i][box.j - 1])
            if box.j < self.rows - 1:
                candidates.append(self.grid[box.i][box.j + 1])
            for candidate in candidates:
                if candidate.type is not TYPES.WALL:
                    box.neighbors.append(candidate)

    def reset(self):
        for box in self.grid.ravel():
            box.type = TYPES.DEFAULT

    def draw(self):
        for box in self.grid.ravel():
            # Draw a box
            pygame.draw.rect(
                self.window,
                box.type.color,
                (
                    box.i * Box.width,
                    box.j * Box.height,
                    Box.width - 2,
                    Box.height - 2,
                ),
            )

        pygame.display.update()

    def __call__(self, x, y):
        i = x // Box.width
        j = y // Box.height
        return self.grid[i][j]


class Dijkstra:
    def __init__(self, game):
        self.game = game
        self.reset()

    def reset(self):
        self.queue = []
        self.path = []
        self.visited = []

    def run(self):
        self.queue.append(TYPES.START.box)
        finished = False

        while len(self.queue) > 0 and not finished:
            self.game.draw()

            current_box = self.queue.pop(0)
            current_box.type = TYPES.VISITED
            self.visited.append(current_box)

            if current_box is TYPES.TARGET.box:
                finished = True

                # Draw Path
                while current_box.prior is not TYPES.START.box:
                    self.path.append(current_box.prior)
                    current_box.prior.type = TYPES.PATH
                    current_box = current_box.prior
            else:
                for neighbor in current_box.neighbors:
                    if neighbor not in self.visited and neighbor not in self.queue:
                        neighbor.prior = current_box
                        neighbor.type = TYPES.QUEUED
                        self.queue.append(neighbor)

        if not finished:
            print("No solution found")

        print("Exiting Dijkstra algorithm")


def main():
    # PYGAME INITIALIZE
    pygame.init()
    window = pygame.display.set_mode((400, 400))

    game = Game(window)
    algorithm = Dijkstra(game)

    done = False
    while not done:
        window.fill((0, 0, 0))

        for event in pygame.event.get():
            x, y = pygame.mouse.get_pos()

            # QUIT WINDOWS
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game.reset()
                    algorithm.reset()

                elif event.key == pygame.K_RETURN:
                    print("Start Algorithm")
                    game.set_neighbors()
                    algorithm.run()

            elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                if pygame.mouse.get_pressed()[0]:
                    game(x, y).type = TYPES.START

            elif pygame.key.get_mods() & pygame.KMOD_CTRL:
                if pygame.mouse.get_pressed()[0]:
                    game(x, y).type = TYPES.TARGET

            elif pygame.mouse.get_pressed()[0]:
                game(x, y).type = TYPES.WALL

            elif pygame.mouse.get_pressed()[2]:
                game(x, y).type = TYPES.DEFAULT

        # DRAW BOXES
        game.draw()

    pygame.quit()


if __name__ == "__main__":
    main()
