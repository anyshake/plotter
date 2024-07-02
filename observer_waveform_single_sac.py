# -*- coding: utf-8 -*-
import obspy
import numpy
import matplotlib.pyplot

window_size = 2
overlap_percent = 86

fig, axs = matplotlib.pyplot.subplots(2, 1, figsize = (12.0, 8.0))
matplotlib.pyplot.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 0.05, wspace = 0)

st_bhz = obspy.read("./2024.183.03.18.27.0083.SHAKE.AS.00.EHZ.D.sac")[0]

for i, st, component in zip(range(1), [st_bhz], ["BHZ"]):
    st = st.copy().detrend("linear")
    axs[i*2].clear()
    axs[i*2+1].clear()
    times = numpy.arange(st.stats.npts) / st.stats.sampling_rate
    waveform_data = st.copy().filter("bandpass", freqmin = 0.1, freqmax = 10.0, zerophase = True).data
    axs[i*2].plot(times, waveform_data, label = component, color = "blue")
    axs[i*2].legend(loc = "upper left")
    axs[i*2].xaxis.set_visible(False)
    axs[i*2].yaxis.set_visible(False)
    axs[i*2].set_xlim([times[0], times[-1]])
    axs[i*2].set_ylim([numpy.min(waveform_data), numpy.max(waveform_data)])
    NFFT = int(st.stats.sampling_rate * window_size)
    noverlap = int(NFFT * (overlap_percent / 100))
    Pxx, freqs, bins, im = axs[i*2+1].specgram(st.copy().filter("highpass", freq = 0.1, zerophase = True).data, NFFT = NFFT, Fs = st.stats.sampling_rate, noverlap = noverlap, cmap = "jet", vmin = 20, vmax = 120)
    axs[i*2+1].set_ylim(0, 15)
    axs[i*2+1].yaxis.set_visible(False)
    axs[i*2+1].xaxis.set_visible(False)

if __name__ == "__main__":
    matplotlib.pyplot.show()
