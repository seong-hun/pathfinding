from itertools import chain

import pygame

window_width = 400
window_height = 400
columns = 20
rows = 20

box_width = window_width // columns
box_height = window_height // rows

# PYGAME INITIALIZE
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()


class Box:
    colormap = {
        "default": (80, 80, 80),
        "queued": (200, 0, 0),
        "visited": (0, 200, 0),
        "path": (0, 0, 200),
        "wall": (120, 120, 120),
        "start": (0, 200, 200),
        "target": (200, 200, 0),
    }

    def __init__(self, i, j):
        self.i = i
        self.j = j

        self.tags = []
        self.neighbors = []
        self.prior = None

    def draw(self):
        color = self.colormap["default"]
        for key, value in self.colormap.items():
            if key in self.tags:
                color = value

        pygame.draw.rect(
            window,
            color,
            (
                self.i * box_width,
                self.j * box_height,
                box_width - 2,
                box_height - 2,
            ),
        )


def main():
    # Create Grid
    box_grid = [[Box(i, j) for j in range(rows)] for i in range(columns)]
    boxes = list(chain.from_iterable(box_grid))

    # Set Neighbours
    for box in boxes:
        if box.i > 0:
            box.neighbors.append(box_grid[box.i - 1][box.j])
        if box.i < columns - 1:
            box.neighbors.append(box_grid[box.i + 1][box.j])
        if box.j > 0:
            box.neighbors.append(box_grid[box.i][box.j - 1])
        if box.j < rows - 1:
            box.neighbors.append(box_grid[box.i][box.j + 1])

    def get_box(x, y):
        i = x // box_width
        j = y // box_height
        return box_grid[i][j]

    queue = []
    path = []

    start_box = box_grid[0][0]
    start_box.tags = ["start", "visited"]
    queue.append(start_box)

    begin_search = False
    searching = True
    target_box = None

    done = False

    while not done:
        window.fill((0, 0, 0))

        for event in pygame.event.get():
            # QUIT WINDOWS
            if event.type == pygame.QUIT:
                done = True

            mouse = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()

            # MOUSE CONTROLS
            if mouse[0]:
                x, y = pygame.mouse.get_pos()

                # Draw Wall
                if keys[pygame.K_w]:
                    get_box(x, y).tags.append("wall")

                # Set Target
                if keys[pygame.K_t]:
                    if target_box is not None:
                        target_box.tags.remove("target")
                    target_box = get_box(x, y)
                    target_box.tags.append("target")

            # Start Algorithm
            if keys[pygame.K_RETURN] and target_box is not None:
                begin_search = True

        if begin_search:
            if len(queue) > 0 and searching:
                current_box = queue.pop(0)
                current_box.tags.append("visited")
                if current_box == target_box:
                    searching = False
                    while current_box.prior != start_box:
                        path.append(current_box.prior)
                        current_box.prior.tags.append("path")
                        current_box = current_box.prior
                else:
                    for neighbor in current_box.neighbors:
                        if (
                            not "queued" in neighbor.tags
                            and not "wall" in neighbor.tags
                        ):
                            neighbor.prior = current_box
                            neighbor.tags.append("queued")
                            queue.append(neighbor)
            else:
                searching = False

        # DRAW BOXES
        for box in boxes:
            box.draw()

        pygame.display.flip()

        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
