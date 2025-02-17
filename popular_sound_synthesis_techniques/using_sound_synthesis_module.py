import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from IPython.display import Audio
from sound_synthesis_module import AdditiveSynthesis, SubtractiveSynthesis, FMSynthesis, WavetableSynthesis, GranularSynthesis, KarplusStrong, SampleSynthesis

# Function to plot a signal
def plot_signal(signal, title="Signal", num_samples=1000):
    """
    Plots the given signal. Displays only the first `num_samples` samples for better visualization.

    Parameters:
    - signal (ndarray): The signal data to plot.
    - title (str): The title of the plot.
    - num_samples (int): Number of samples to plot (default is 1000).
    """
    plt.figure(figsize=(10, 4))
    plt.plot(signal[:num_samples])  # Plot first `num_samples` samples
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

# Additive Synthesis
frequencies = [440, 880, 1320]  # Example frequencies (A4, A5, A6)
amplitudes = [0.5, 0.3, 0.2]  # Amplitudes for each frequency
additive_synth = AdditiveSynthesis(frequencies, amplitudes, duration=2.0)
additive_signal = additive_synth.generate_signal()

# Plot Additive Synthesis Signal
plot_signal(additive_signal, title="Additive Synthesis Signal")

# Save the Additive Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\additive_synth.wav", 44100, np.int16(additive_signal * 32767))  # Convert signal to 16-bit PCM

# Subtractive Synthesis
subtractive_synth = SubtractiveSynthesis(frequency=440, duration=2.0, cutoff=500)
subtractive_signal = subtractive_synth.generate_signal()

# Plot Subtractive Synthesis Signal
plot_signal(subtractive_signal, title="Subtractive Synthesis Signal")

# Save the Subtractive Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\subtractive_synth.wav", 44100, np.int16(subtractive_signal * 32767))  # Convert signal to 16-bit PCM

# FM Synthesis
fm_synth = FMSynthesis(carrier_freq=440, modulator_freq=2, modulation_index=5, duration=2.0)
fm_signal = fm_synth.generate_signal()

# Plot FM Synthesis Signal
plot_signal(fm_signal, title="FM Synthesis Signal")

# Save the FM Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\fm_synth.wav", 44100, np.int16(fm_signal * 32767))  # Convert signal to 16-bit PCM

# Wavetable Synthesis
wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1000))  # Simple sine wave wavetable
wavetable_synth = WavetableSynthesis(wavetable=wavetable, frequency=440, duration=2.0)
wavetable_signal = wavetable_synth.generate_signal()

# Plot Wavetable Synthesis Signal
plot_signal(wavetable_signal, title="Wavetable Synthesis Signal")

# Save the Wavetable Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\wavetable_synth.wav", 44100, np.int16(wavetable_signal * 32767))  # Convert signal to 16-bit PCM

# Granular Synthesis
granular_signal = np.random.rand(44100)  # Random signal as an example
granular_synth = GranularSynthesis(signal=granular_signal, grain_size=500, overlap=0.5)
granular_signal = granular_synth.generate_signal()

# Plot Granular Synthesis Signal
plot_signal(granular_signal, title="Granular Synthesis Signal")

# Save the Granular Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\granular_synth.wav", 44100, np.int16(granular_signal * 32767))  # Convert signal to 16-bit PCM

# Karplus-Strong Synthesis
karplus_strong_synth = KarplusStrong(frequency=440, duration=2.0)
karplus_strong_signal = karplus_strong_synth.generate_signal()

# Plot Karplus-Strong Synthesis Signal
plot_signal(karplus_strong_signal, title="Karplus-Strong Synthesis Signal")

# Save the Karplus-Strong Synthesis Signal to a WAV file
write(r"popular_sound_synthesis_techniques\sound_synthesis_module_usage_sounds\karplus_strong_synth.wav", 44100, np.int16(karplus_strong_signal * 32767))  # Convert signal to 16-bit PCM




