import pygame
import matplotlib as mpl


class Cell(pygame.sprite.Sprite):
    """
    A class to manage each cell in the mesh.
    """
    COLOR_MAP = mpl.colormaps["bwr"]

    def __init__(self, loc: tuple, width: int,
                 height: int, value: float = 1.0):
        """
        The constructor of the `Cell` class.

        Parameters
        ----------
        loc : tuple
            The location of the top left corner of the rectangle of the cell.
        width : int
            The width of the rectangle of the cell.
        height : int
            The height of the rectangle of the cell.
        value : float, optional
            The value of the displacement in this cell, by default 1.0.
        """
        pygame.sprite.Sprite.__init__(self)
        self.__width = width
        self.__height = height
        self.__loc = loc
        self.image = pygame.Surface([self.__width, self.__height])
        self.rect = self.image.get_rect(topleft=self.__loc)

        self.__value = value
        self.update(value)

    def update(self, value: float):
        """
        Updates the value of this cell and its image
        """
        if self.__value != value:
            self.__value = value
            self.__color = [
                int(255 * i) for i in Cell.COLOR_MAP(self.__value)]
            self.image.fill(self.__color)
