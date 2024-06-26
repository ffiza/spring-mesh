import pygame
import sys
import numpy as np
import time

from cell import Cell


class MyGroup(pygame.sprite.Group):
    def update(self, values):
        for i, sprite in enumerate(self.sprites()):
            sprite.update(values[i])


class BlackScreen:
    SCREEN_SIZE: tuple = (800, 600)
    COLOR_DEPTH: int = 32
    WINDOW_NAME: str = "Black Screen"
    FONT_SIZE: int = 12
    FPS: int = 60
    BACKGROUND_COLOR: str = "gray10"

    def __init__(self) -> None:
        pygame.init()
        self.time = time.time()
        self.screen = pygame.display.set_mode(
            size=BlackScreen.SCREEN_SIZE,
            flags=pygame.NOFRAME,
            depth=BlackScreen.COLOR_DEPTH)
        pygame.display.set_caption(BlackScreen.WINDOW_NAME)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, BlackScreen.FONT_SIZE)

        sprite_1 = Cell(loc=(100, 100), width=50, height=50, value=1.0)
        sprite_2 = Cell(loc=(400, 100), width=50, height=50, value=0.5)
        self.sprites = MyGroup()
        self.sprites.add(sprite_1, sprite_2)

    @staticmethod
    def _quit() -> None:
        """
        Quit the animation and close the window.
        """
        pygame.quit()
        sys.exit()

    def _handle_user_events(self) -> None:
        """
        Handle the user inputs.
        """
        for event in pygame.event.get():
            # Quitting
            if event.type == pygame.QUIT:
                BlackScreen._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                BlackScreen._quit()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:  # Main game loop
            self._handle_user_events()

            self.sprites.update([np.cos(self.time), np.sin(self.time)])
            self.sprites.draw(self.screen)

            pygame.display.flip()

            self.dt = self.clock.tick(BlackScreen.FPS) / 1000
            self.time += self.dt


if __name__ == "__main__":
    animation = BlackScreen()
    animation.run()