import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from functions.decorators import plotmethod


class FilterWrapper(object):
    def __init__(self, name, critical_value):
        self.critical_value = critical_value
        self.name = name

    @abstractmethod
    def apply(self, time, data, plot_fl=False):
        pass

    @plotmethod
    def plot(self, time, data, apply_plot_fl=False, fig=None, range=None, ax_labels=('Time', 'Signal Amplitude')):
        filtered_data = self.apply(time, data, plot_fl=apply_plot_fl)

        if range and isinstance(range, slice):
            data = data[range]
            filtered_data = filtered_data[range]

        if not fig:
            fig = plt.figure()
        plt.plot(time, data, label='Raw', color='silver')
        plt.plot(time, filtered_data, label='Filtered')

        plt.legend()


###################################################################################
#                                                                                 #
#                                 Gated Filter                                    #
#                                                                                 #
###################################################################################


class GatedFilter(FilterWrapper):
    def __init__(self, critical_amplitude):
        super().__init__('Gated Filter', critical_amplitude)

    def apply(self, time, data, plot_fl=False):
        spectrum = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), time[1] - time[0])
        amplitudes = np.abs(spectrum) / np.max(np.abs(spectrum))
        sp_gated = spectrum.copy()
        for i, amp in enumerate(amplitudes):
            if amp < self.critical_value:
                sp_gated[i] = 0

        # Apply gate filter to data
        gated_data = np.fft.ifft(sp_gated)

        if plot_fl:
            # Plots of the gated spectrum, i.e. frequencies with an amplitude below a certain cutoff are rejected.
            plt.figure()
            plt.subplot(211)
            plt.semilogy(freq, np.abs(spectrum) / np.max(np.abs(spectrum)), 'x', label='FFT Spectrum')
            plt.axhline(y=self.critical_value, linewidth=1.0, linestyle='dotted', color='red')
            plt.ylabel('Amplitude')
            plt.grid(which='both', axis='both')
            plt.xlim(0, 1500)
            plt.ylim(1e-6, 1)
            plt.legend()

            plt.subplot(212)
            plt.semilogy(freq, np.abs(sp_gated) / np.max(np.abs(sp_gated)), 'x', label='Gated Spectrum')
            plt.axhline(y=self.critical_value, linewidth=1.0, linestyle='dotted', color='red')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(which='both', axis='both')
            plt.xlim(0, 1500)
            plt.ylim(1e-6, 1)
            plt.legend()

            # Plot of inverse-fft'd gated spectrum compared to the raw_signal
            plt.figure()
            plt.plot(time, data, label='Raw signal', color='silver')
            plt.plot(time, gated_data, label='Gated signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Current (A)')
            plt.legend()

            plt.show()

        return gated_data


###################################################################################
#                                                                                 #
#                               Low Pass Filter                                   #
#                                                                                 #
###################################################################################

class LowPassFilter(FilterWrapper):
    _DEFAULT_KWARGS = {'analog': False}
    _DEFAULT_ARGS = [6]

    def __init__(self, critical_frequency, filter_handle=None,
                 filter_args=_DEFAULT_ARGS, filter_kwargs=_DEFAULT_KWARGS):
        super().__init__('Low Pass Filter', critical_frequency)
        self.filter_args = filter_args
        self.filter_kwargs = filter_kwargs
        if not filter_handle:
            self.filter_handle = sig.butter
        else:
            self.filter_handle = filter_handle

    def apply(self, time, data, plot_fl=False):
        spectrum = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), time[1] - time[0])

        nyq = 1 / (2 * (time[1] - time[0]))
        crit_freq_norm = self.critical_value / nyq
        if self.filter_handle is sig.butter:
            b, a = self.filter_handle(*self.filter_args, crit_freq_norm, **self.filter_kwargs)
        else:
            b, a = self.filter_handle(*self.filter_args, **self.filter_kwargs)

        # Apply filter to raw data
        filtered_data = sig.filtfilt(b, a, data)

        if plot_fl and self.filter_handle is sig.butter:
            w, h = sig.freqz(b, a)

            plt.figure()
            plt.plot((nyq * w) / np.pi, abs(h))
            plt.title('Butterworth filter frequency response')
            plt.xlabel('Frequency [radians / second]')
            plt.ylabel('Amplitude [dB]')
            plt.xlim(0, self.critical_value * 2)
            plt.grid(which='both', axis='both')
            plt.axvline(self.critical_value, color='green')  # cutoff frequency

            # Plot spectrogram of the data before and after filtering
            sample_freq = 2 * nyq

            plt.figure()
            plt.subplot(121)
            ff, tt, Sxx = sig.spectrogram(data, sample_freq)
            Sxx_log = np.log(Sxx)
            plt.pcolormesh(tt, ff, Sxx_log)
            cbar = plt.colorbar()
            cbar.set_label('Log(Amplitude)')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')

            plt.subplot(122)
            ff, tt, Sxx = sig.spectrogram(filtered_data, sample_freq)
            Sxx_log = np.log(Sxx)
            plt.pcolormesh(tt, ff, Sxx_log)
            cbar = plt.colorbar()
            cbar.set_label('Log(Amplitude)')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')

            # Plot a side-by-side comparison of the FFT spectrum before and after applying the filter, along with
            # the filter shape and cut-off frequency.
            plt.figure()
            ax1 = plt.subplot(211)
            plt.semilogy(freq, np.abs(spectrum) / np.max(np.abs(spectrum)), label='FFT Spectrum')
            plt.ylim(1e-7, 1.1)
            plt.xlim(0, self.critical_value * 2)
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.grid(which='both', axis='both')

            ax2 = ax1.twinx()
            plt.plot((nyq * w) / np.pi, abs(h), color='orange', label='Filter')
            ax2.set_ylim(1e-7, 1.01)
            plt.ylabel('Filter Amplitude')
            plt.axvline(self.critical_value, color='green', label='Cutoff Freq')  # cutoff frequency

            sp_postfilter = np.fft.fft(filtered_data)

            ax1 = plt.subplot(212)
            plt.semilogy(freq, np.abs(sp_postfilter) / np.max(np.abs(sp_postfilter)))
            plt.ylim(1e-7, 1.1)
            plt.xlim(0, self.critical_value * 2)
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude')
            plt.grid(which='both', axis='both')

            ax2 = ax1.twinx()
            ax2.set_ylim(1e-7, 1.01)
            plt.plot((nyq * w) / np.pi, abs(h), color='orange')
            plt.ylabel('Filter Amplitude')
            plt.axvline(self.critical_value, color='green')  # cutoff frequency

            plt.show()

        return filtered_data

