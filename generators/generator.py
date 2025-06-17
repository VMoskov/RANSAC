from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def _apply_salt_pepper_noise(self):
        pass

    @abstractmethod
    def visualize(self):
        pass