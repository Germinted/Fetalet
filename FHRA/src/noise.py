"""
Noise generation module for FHR signals.
Implements various types of noise that can be added to fetal heart rate data.
"""

import numpy as np


class NoiseGenerator:
    """Base class for noise generators."""

    def __init__(self, params=None):
        """
        Initialize noise generator with parameters.

        Args:
            params (dict): Dictionary of noise-specific parameters
        """
        self.params = params or {}

    def add_noise(self, signal):
        """
        Add noise to the signal.

        Args:
            signal (np.ndarray): Input signal of shape (channels, length)

        Returns:
            np.ndarray: Noisy signal
        """
        raise NotImplementedError


class GaussianNoise(NoiseGenerator):
    """Gaussian noise generator."""

    def __init__(self, params=None):
        default_params = {'noise_level': 0.1}
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_noise(self, signal):
        """Add Gaussian noise to the FHR channel (index 0)."""
        noise = np.random.normal(
            0,
            self.params['noise_level'],
            signal[0].shape
        )
        noisy_signal = signal.copy()
        noisy_signal[0] = noisy_signal[0] + noise
        return noisy_signal


class SaltPepperNoise(NoiseGenerator):
    """Salt and pepper noise generator."""

    def __init__(self, params=None):
        default_params = {
            'amount': 0.05,
            'salt_vs_pepper': 0.5
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_noise(self, signal):
        """Add salt and pepper noise to the FHR channel."""
        noisy_signal = signal.copy()
        fhr_signal = signal[0]

        # Generate random mask
        mask = np.random.rand(*fhr_signal.shape) < self.params['amount']

        # Add salt (positive spikes)
        salt_mask = mask & (np.random.rand(*fhr_signal.shape) < self.params['salt_vs_pepper'])
        noisy_signal[0][salt_mask] = np.max(fhr_signal) + 10

        # Add pepper (negative spikes)
        pepper_mask = mask & ~salt_mask
        noisy_signal[0][pepper_mask] = np.max(0, np.min(fhr_signal) - 10)

        return noisy_signal


class BaselineDrift(NoiseGenerator):
    """Baseline drift noise generator."""

    def __init__(self, params=None):
        default_params = {
            'frequency': 0.01,  # Hz
            'amplitude': 5
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_noise(self, signal):
        """Add baseline drift to the FHR channel."""
        t = np.arange(0, signal[0].shape[0])
        drift = self.params['amplitude'] * np.sin(
            2 * np.pi * self.params['frequency'] * t
        )

        noisy_signal = signal.copy()
        noisy_signal[0] = noisy_signal[0] + drift
        return noisy_signal


class PowerLineInterference(NoiseGenerator):
    """Power line interference noise generator."""

    def __init__(self, params=None):
        default_params = {
            'frequency': 50,  # Hz (50 or 60)
            'amplitude': 1.0,
            'sampling_rate': 4  # samples per second
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_noise(self, signal):
        """Add power line interference to the FHR channel."""
        t = np.arange(0, signal[0].shape[0]) / self.params['sampling_rate']
        interference = self.params['amplitude'] * np.sin(
            2 * np.pi * self.params['frequency'] * t
        )

        # Ensure same length as signal
        interference = interference[:signal[0].shape[0]]

        noisy_signal = signal.copy()
        noisy_signal[0] = noisy_signal[0] + interference
        return noisy_signal


class ImpulseNoise(NoiseGenerator):
    """Random impulse noise generator."""

    def __init__(self, params=None):
        default_params = {
            'probability': 0.02,
            'amplitude': 10
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def add_noise(self, signal):
        """Add random impulse noise to the FHR channel."""
        # Generate random impulses
        mask = np.random.rand(*signal[0].shape) < self.params['probability']
        impulses = np.random.choice(
            [-1, 1],
            size=signal[0].shape
        ) * self.params['amplitude'] * mask

        noisy_signal = signal.copy()
        noisy_signal[0] = noisy_signal[0] + impulses
        return noisy_signal


class NoiseFactory:
    """Factory class for creating noise generators."""

    NOISE_TYPES = {
        'gaussian': GaussianNoise,
        'salt_pepper': SaltPepperNoise,
        'baseline_drift': BaselineDrift,
        'power_line': PowerLineInterference,
        'impulse': ImpulseNoise
    }

    @classmethod
    def create_noise_generator(cls, noise_type, params=None):
        """
        Create a noise generator of the specified type.

        Args:
            noise_type (str): Type of noise
            params (dict): Parameters for the noise generator

        Returns:
            NoiseGenerator: Instance of the specified noise generator
        """
        if noise_type not in cls.NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return cls.NOISE_TYPES[noise_type](params)

    @classmethod
    def apply_multiple_noises(cls, signal, noise_types, noise_params=None):
        """
        Apply multiple types of noise to a signal.

        Args:
            signal (np.ndarray): Input signal of shape (channels, length)
            noise_types (list): List of noise types to apply
            noise_params (dict): Parameters for each noise type

        Returns:
            np.ndarray: Signal with multiple noises applied
        """
        noisy_signal = signal.copy()
        noise_params = noise_params or {}

        for noise_type in noise_types:
            params = noise_params.get(noise_type, {})
            generator = cls.create_noise_generator(noise_type, params)
            noisy_signal = generator.add_noise(noisy_signal)

        return noisy_signal