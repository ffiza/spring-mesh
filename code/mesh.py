import pygame


class Mesh(pygame.sprite.Group):
    def update(self, values):
        for i, sprite in enumerate(self.sprites()):
            sprite.update(values[i])
