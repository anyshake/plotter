# -*- coding: utf-8 -*-
import os
import json
import time
import obspy
import numpy
import asyncio
import threading
import websockets
import matplotlib.pyplot
import matplotlib.animation

station_net = "AS" # Station network code
station_stn = "SHAKE" # Station station code
station_loc = "00" # Station location code
channel_prefix = "E" # Station channel prefix code E.g. EHx == E, BHx == B
station_address = "127.0.0.1:8073" # Observer address

time_span = 120 # Time span in seconds
refresh_time = 1000 # Refresh time in milliseconds
window_size = 2 # Spectrogram window size in seconds
overlap_percent = 86 # Spectrogram overlap in percent
spectrogram_power_range = [20, 120] # Spectrogram power range in dB

fig, axs = matplotlib.pyplot.subplots(6, 1, num="Observer Waveform", figsize=(12.0, 8.0))
matplotlib.pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

def resample_trace(trace, target_sampling_rate):
    if trace.stats.sampling_rate != target_sampling_rate:
        max_sampling_rate = max(trace.stats.sampling_rate, target_sampling_rate)
        trace = trace.copy().resample(max_sampling_rate)
    return trace

def make_trace(channel, sps, counts_list, timestamp):
    trace = obspy.core.Trace(data=numpy.ma.MaskedArray(counts_list, dtype=numpy.float64))
    trace.stats.network = station_net
    trace.stats.station = station_stn
    trace.stats.location = station_loc
    trace.stats.channel = channel
    trace.stats.sampling_rate = sps
    trace.stats.starttime = obspy.UTCDateTime(timestamp)
    return trace

async def get_data():
    global bhe_data, bhn_data, bhz_data
    while True:
        try:
            async with websockets.connect(f"ws://{station_address}/api/v1/socket") as websocket:
                while True:
                    data = json.loads(await websocket.recv())
                    timestamp = data["ts"] / 1000
                    bhe_data = make_trace(f"{channel_prefix}HE", len(data["ehe"]), data["ehe"], timestamp)
                    bhn_data = make_trace(f"{channel_prefix}HN", len(data["ehn"]), data["ehn"], timestamp)
                    bhz_data = make_trace(f"{channel_prefix}HZ", len(data["ehz"]), data["ehz"], timestamp)
        except Exception as e:
            print(e)
            time.sleep(0.5)
            continue

def update(frame):
    try:
        st_bhe = bhe_stream.copy().detrend("linear")
        st_bhn = bhn_stream.copy().detrend("linear")
        st_bhz = bhz_stream.copy().detrend("linear")
        for i, st, component in zip(range(3), [st_bhe, st_bhn, st_bhz], [f"{channel_prefix}HE", f"{channel_prefix}HN", f"{channel_prefix}HZ"]):
            axs[i*2].clear()
            axs[i*2+1].clear()
            times = numpy.arange(st.stats.npts) / st.stats.sampling_rate
            waveform_data = st.copy().filter("bandpass", freqmin=0.1, freqmax=10.0, zerophase=True).data
            axs[i*2].plot(times, waveform_data, label=component, color="blue")
            axs[i*2].legend(loc="upper left")
            axs[i*2].xaxis.set_visible(False)
            axs[i*2].yaxis.set_visible(False)
            axs[i*2].set_xlim([times[0], times[-1]])
            axs[i*2].set_ylim([numpy.min(waveform_data), numpy.max(waveform_data)])
            NFFT = int(st.stats.sampling_rate * window_size)
            noverlap = int(NFFT * (overlap_percent / 100))
            Pxx, freqs, bins, im = axs[i*2+1].specgram(st.copy().filter("highpass", freq=0.1, zerophase=True).data, NFFT=NFFT, Fs=st.stats.sampling_rate, noverlap=noverlap, cmap="jet", vmin=spectrogram_power_range[0], vmax=spectrogram_power_range[1])
            axs[i*2+1].set_ylim(0, 15)
            axs[i*2+1].yaxis.set_visible(False)
            axs[i*2+1].xaxis.set_visible(False)

        # Update bhe_stream
        new_samples = int(bhe_data.stats.sampling_rate)
        if len(bhe_stream.data) >= bhe_stream.stats.sampling_rate * time_span:
            bhe_stream.data = numpy.delete(bhe_stream.data, slice(new_samples))
        bhe_stream.data = numpy.concatenate((bhe_stream.data, resample_trace(bhe_data, bhe_stream.stats.sampling_rate).data))
        bhe_stream.stats.starttime = bhe_stream.stats.starttime + 1.0

        # Update bhn_stream
        new_samples = int(bhn_data.stats.sampling_rate)
        if len(bhn_stream.data) >= bhn_stream.stats.sampling_rate * time_span:
            bhn_stream.data = numpy.delete(bhn_stream.data, slice(new_samples))
        bhn_stream.data = numpy.concatenate((bhn_stream.data, resample_trace(bhn_data, bhn_stream.stats.sampling_rate).data))
        bhn_stream.stats.starttime = bhn_stream.stats.starttime + 1.0

        # Update bhz_stream
        new_samples = int(bhz_data.stats.sampling_rate)
        if len(bhz_stream.data) >= bhz_stream.stats.sampling_rate * time_span:
            bhz_stream.data = numpy.delete(bhz_stream.data, slice(new_samples))
        bhz_stream.data = numpy.concatenate((bhz_stream.data, resample_trace(bhz_data, bhz_stream.stats.sampling_rate).data))
        bhz_stream.stats.starttime = bhz_stream.stats.starttime + 1.0

    except Exception as e:
        print(f"Error plotting data: {e}")

if __name__ == "__main__":
    thread1 = threading.Thread(target=asyncio.run, args=(get_data(),))
    thread1.start()
    time.sleep(3)
    bhe_stream = bhe_data.copy()
    bhn_stream = bhn_data.copy()
    bhz_stream = bhz_data.copy()
    for _ in range(time_span):
        bhe_stream.data = numpy.concatenate((bhe_stream.data, resample_trace(bhe_data, bhe_stream.stats.sampling_rate).data))
        bhn_stream.data = numpy.concatenate((bhn_stream.data, resample_trace(bhn_data, bhn_stream.stats.sampling_rate).data))
        bhz_stream.data = numpy.concatenate((bhz_stream.data, resample_trace(bhz_data, bhz_stream.stats.sampling_rate).data))
        time.sleep(1)
    ani = matplotlib.animation.FuncAnimation(fig, update, interval=refresh_time, cache_frame_data=False)
    matplotlib.pyplot.show()
    os._exit(0)
