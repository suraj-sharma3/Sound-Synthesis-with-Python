import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from IPython.display import Audio

# Additive Synthesis class - Generates signals by summing sine waves at different frequencies and amplitudes
class AdditiveSynthesis:
    def __init__(self, frequencies, amplitudes, duration=1.0, sample_rate=44100):
        """
        Initialize the AdditiveSynthesis object with specified frequencies, amplitudes, duration, and sample rate.

        Parameters:
        frequencies (list): List of frequencies of the sine waves to sum.
        amplitudes (list): List of amplitudes corresponding to each frequency.
        duration (float): Duration of the signal in seconds (default is 1.0).
        sample_rate (int): Sample rate of the audio (default is 44100).
        """
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.duration = duration
        self.sample_rate = sample_rate
    
    def generate_signal(self):
        """
        Generate an additive synthesis signal by summing sine waves.

        Returns:
        numpy.ndarray: The generated audio signal.
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        signal_data = np.zeros_like(t)
        for f, a in zip(self.frequencies, self.amplitudes):
            signal_data += a * np.sin(2 * np.pi * f * t)
        return signal_data

# Subtractive Synthesis class - Generates signals by filtering a sawtooth wave
class SubtractiveSynthesis:
    def __init__(self, frequency, duration=1.0, sample_rate=44100, cutoff=1000):
        """
        Initialize the SubtractiveSynthesis object with specified frequency, duration, sample rate, and cutoff.

        Parameters:
        frequency (float): Frequency of the base wave.
        duration (float): Duration of the signal in seconds (default is 1.0).
        sample_rate (int): Sample rate of the audio (default is 44100).
        cutoff (float): Cutoff frequency for the low-pass filter (default is 1000).
        """
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.cutoff = cutoff

    def generate_signal(self):
        """
        Generate a subtractive synthesis signal by filtering a sawtooth wave.

        Returns:
        numpy.ndarray: The generated audio signal after filtering.
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        sawtooth_wave = 0.5 * (1 - np.mod(t * self.frequency, 1))  # Generate sawtooth wave
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff / nyquist
        b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)  # Design low-pass filter
        filtered_signal = signal.filtfilt(b, a, sawtooth_wave)  # Apply the filter
        return filtered_signal

# FM Synthesis class - Frequency modulation synthesis to create complex waveforms
class FMSynthesis:
    def __init__(self, carrier_freq, modulator_freq, modulation_index, duration=1.0, sample_rate=44100):
        """
        Initialize the FMSynthesis object with carrier frequency, modulator frequency, modulation index, duration, and sample rate.

        Parameters:
        carrier_freq (float): Frequency of the carrier wave.
        modulator_freq (float): Frequency of the modulator wave.
        modulation_index (float): Modulation index that controls the depth of modulation.
        duration (float): Duration of the signal in seconds (default is 1.0).
        sample_rate (int): Sample rate of the audio (default is 44100).
        """
        self.carrier_freq = carrier_freq
        self.modulator_freq = modulator_freq
        self.modulation_index = modulation_index
        self.duration = duration
        self.sample_rate = sample_rate

    def generate_signal(self):
        """
        Generate a signal using frequency modulation synthesis.

        Returns:
        numpy.ndarray: The generated audio signal.
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        modulator = np.sin(2 * np.pi * self.modulator_freq * t) * self.modulation_index
        carrier = np.sin(2 * np.pi * self.carrier_freq * t + modulator)  # FM signal
        return carrier

# Wavetable Synthesis class - Generate audio by indexing a wavetable
class WavetableSynthesis:
    def __init__(self, wavetable, frequency, duration=1.0, sample_rate=44100):
        """
        Initialize the WavetableSynthesis object with a wavetable, frequency, duration, and sample rate.

        Parameters:
        wavetable (numpy.ndarray): The wavetable containing waveform samples.
        frequency (float): Frequency of the output signal.
        duration (float): Duration of the signal in seconds (default is 1.0).
        sample_rate (int): Sample rate of the audio (default is 44100).
        """
        self.wavetable = wavetable
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate

    def generate_signal(self):
        """
        Generate a signal by sampling a wavetable at a given frequency.

        Returns:
        numpy.ndarray: The generated audio signal using wavetable synthesis.
        """
        num_samples = int(self.sample_rate * self.duration)
        phase = np.linspace(0, 1, num_samples) * self.frequency  # Calculate phase over time
        phase = np.mod(phase, 1)  # Wrap phase to stay within bounds
        indices = (phase * len(self.wavetable)).astype(int)  # Generate indices into the wavetable
        return self.wavetable[indices]

# Granular Synthesis class - Create audio by concatenating overlapping grains from a sample
class GranularSynthesis:
    def __init__(self, signal, grain_size, overlap, sample_rate=44100):
        """
        Initialize the GranularSynthesis object with a signal, grain size, overlap, and sample rate.

        Parameters:
        signal (numpy.ndarray): The audio signal to process.
        grain_size (int): Size of each grain (in samples).
        overlap (float): Overlap factor between grains.
        sample_rate (int): Sample rate of the audio (default is 44100).
        """
        self.signal = signal
        self.grain_size = grain_size
        self.overlap = overlap
        self.sample_rate = sample_rate

    def generate_signal(self):
        """
        Generate a signal using granular synthesis by concatenating grains.

        Returns:
        numpy.ndarray: The generated audio signal with grains concatenated.
        """
        num_samples = len(self.signal)
        grains = []
        for start in range(0, num_samples - self.grain_size, int(self.grain_size * (1 - self.overlap))):
            grain = self.signal[start:start + self.grain_size]  # Extract grain
            grains.append(grain)
        return np.concatenate(grains)  # Concatenate all grains

# Karplus-Strong Synthesis class - Physical modeling synthesis to create plucked string sounds
class KarplusStrong:
    def __init__(self, frequency, duration=1.0, sample_rate=44100):
        """
        Initialize the KarplusStrong object with frequency, duration, and sample rate.

        Parameters:
        frequency (float): Frequency of the plucked string sound.
        duration (float): Duration of the signal in seconds (default is 1.0).
        sample_rate (int): Sample rate of the audio (default is 44100).
        """
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate

    def generate_signal(self):
        """
        Generate a signal using the Karplus-Strong algorithm for plucked string sounds.

        Returns:
        numpy.ndarray: The generated plucked string sound.
        """
        num_samples = int(self.sample_rate * self.duration)
        delay_length = int(self.sample_rate / self.frequency)  # Length of the delay line
        buffer = np.random.rand(delay_length) * 2 - 1  # Random initial buffer values
        signal_data = np.zeros(num_samples)
        for i in range(num_samples):
            signal_data[i] = buffer[i % delay_length]  # Generate the output signal
            buffer[i % delay_length] = 0.5 * (buffer[i % delay_length] + buffer[(i + 1) % delay_length])  # Feedback loop
        return signal_data


