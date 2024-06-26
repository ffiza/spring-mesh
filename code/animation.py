import pygame
import sys
import yaml
import pandas as pd
import argparse
import pickle
import numpy as np

from cell import Cell
from mesh import Mesh

from ui.debugger import Debugger
from ui.indicator_bar import IndicatorBar
from ui.text import Text


class Animation:
    def __init__(self, data: np.ndarray, time: np.ndarray) -> None:
        """
        The constructor for the Animation class.

        Parameters
        ----------
        data : np.ndarray
            The data to animate.
        time : np.ndarray
            The time of each snapshot.
        """
        pygame.init()
        self.config = yaml.safe_load(open("configs/global.yml"))
        self.running = bool(self.config["ANIMATION_STARTUP_RUN_STATE"])
        self.debugging = bool(self.config["ANIMATION_STARTUP_DEBUG_STATE"])
        self.debugger = Debugger()
        self.data = data
        self.time = time
        self.max_time = self.time[-1]
        self.idx = 0  # Current simulation snapshot

        # Normalize data to the [0, 1] range
        target = np.max([np.abs(self.data.max()), np.abs(self.data.min())])
        self.data = (self.data + target) / (target + target)
        self.data = self.data * 5 - (5 / 2 - 0.5)

        # Setup window
        self.screen = pygame.display.set_mode(
            (self.config["SCREEN_WIDTH"], self.config["SCREEN_HEIGHT"]),
            flags=pygame.FULLSCREEN, depth=32)
        pygame.display.set_caption(self.config["WINDOW_NAME"])

        # Setup PyGame clock
        self.clock = pygame.time.Clock()

        # Setup fonts
        font = self.config["FONT"]
        self.font = pygame.font.Font(f"fonts/{font}.ttf", 30)

        # Define geometrical quantities for the animation
        self.ny, self.nx = self.data[0].shape
        self.box_width = np.min([
            self.config["SCREEN_WIDTH"]
            * (1 - 2 * self.config["BORDER_SPACE_FRAC"]) / self.nx,
            self.config["SCREEN_HEIGHT"]
            * (1 - 2 * self.config["BORDER_SPACE_FRAC"]) / self.ny
        ])
        self.x0 = (self.config["SCREEN_WIDTH"] - self.box_width * self.nx) / 2
        self.y0 = (self.config["SCREEN_HEIGHT"] - self.box_width * self.ny) / 2

        # Create sprites and sprite group
        self.mesh = Mesh()
        for i in range(self.nx):
            for j in range(self.ny):
                sprite = Cell(loc=(self.x0 + i * self.box_width,
                                   self.y0 + j * self.box_width),
                              width=self.box_width + 1,
                              height=self.box_width + 1)
                # The +1 in the previous lines fix minor visualization issues
                self.mesh.add(sprite)
        self.mesh.update(list(self.data[0].T.flatten()))

        # Setup time indicator bar and text
        self.time_bar = IndicatorBar(
            0,
            self.config["SCREEN_HEIGHT"] - self.config["TIME_BAR_HEIGHT"],
            self.config["SCREEN_WIDTH"],
            self.config["TIME_BAR_HEIGHT"],
            self.config["INDICATORS_COLOR"], "horizontal")
        ind_x0 = self.config["TEXT_START_FRAC"] * self.config["SCREEN_WIDTH"]
        self.time_text = Text(
            (ind_x0, self.config["SCREEN_HEIGHT"] - 1 * ind_x0),
            self.font, f"{self.time[0]:.2f}",
            self.config["INDICATORS_COLOR"])

        # Setup pause text
        self.pause_text = Text(
            (self.config["SCREEN_WIDTH"] - 0.05 * self.config["SCREEN_HEIGHT"],
             0.05 * self.config["SCREEN_HEIGHT"]),
            self.font, "Paused",
            self.config["INDICATORS_COLOR"], "topright")

    @staticmethod
    def _quit() -> None:
        """
        This method quits the animation and closes the window.
        """
        pygame.quit()
        sys.exit()

    def _check_events(self) -> None:
        """
        Handle the events loop.
        """
        for event in pygame.event.get():
            # Quitting
            if event.type == pygame.QUIT:
                self._quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit()

            # Pause/Unpause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self._switch_pause()

            # Reset simulation
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._reset_animation()

            # Enable debugging
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.debugging = not self.debugging

    def _switch_pause(self) -> None:
        self.running = not self.running
        self.pause_text.set_active(not self.pause_text.get_active())

    def _reset_animation(self) -> None:
        self.idx = 0
        if self.running:
            self._switch_pause()

    def run(self) -> None:
        """
        Run the main animation loop.
        """

        while True:
            self._check_events()

            if self.idx >= len(self.data):
                self._reset_animation()

            self.mesh.update(self.data[self.idx].T.flatten())
            self.time_bar.set_value(self.time[self.idx] / self.max_time)
            self.time_text.set_value(f"Time: {self.time[self.idx]:.1f} s")

            self.screen.fill(self.config["BACKGROUND_COLOR"])
            self.mesh.draw(self.screen)
            self.time_text.draw(self.screen)
            self.time_bar.draw(self.screen)
            self.pause_text.draw(self.screen)

            if self.debugging:
                self.debugger.render(
                    msgs=[f"{int(self.clock.get_fps())} FPS",],
                    screen=self.screen)

            if self.running:
                self.idx += 1

            self.clock.tick(self.config["FPS"])
            pygame.display.update()


def main():
    # Get simulation name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation", type=str, required=True,
        help="The simulation to animate.")
    args = parser.parse_args()

    # Load data file
    pkl_file = open(f"results/{args.simulation}/data.pkl", "rb")
    data = pickle.load(pkl_file)
    pkl_file.close()
    time = pd.read_csv(
        f"results/{args.simulation}/time.csv")["Time"].to_numpy()

    # Run the PyGame animation
    animation = Animation(data=data, time=time)
    animation.run()


if __name__ == "__main__":
    main()
