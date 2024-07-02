# -*- coding: utf-8 -*-
import os
import json
import time
import pytz
import obspy
import numpy
import requests
import datetime
import matplotlib.pyplot
import matplotlib.animation

hours = 6
ts_offset = 0
show_interval = 60
update_interval = 30
sampling_rate = 100
scaling_range = 1000000

def make_trace(channel, sps, counts_list, timestamp):
    trace = obspy.core.Trace(data = numpy.ma.MaskedArray(counts_list, dtype = numpy.int32))	
    trace.stats.network = "AS"
    trace.stats.station = "SHAKE"
    trace.stats.location = "00"
    trace.stats.channel = channel
    trace.stats.sampling_rate = sps
    trace.stats.starttime = obspy.UTCDateTime(timestamp)
    return trace

def get_time():
    six_hours_ago = datetime.datetime.now() - datetime.timedelta(hours = hours)
    six_hours_ago = six_hours_ago.replace(minute = 0, second = 0, microsecond = 0)
    timestamp = int(six_hours_ago.timestamp() * 1000)
    return timestamp

def get_data(start_time, end_time, max_interval = 3600000):
    data = {"BHZ": []}
    while start_time < end_time:
        current_end_time = min(start_time + max_interval, end_time)
        payload = {"start": start_time, "end": current_end_time, "format": "json"}
        response = requests.post("http://127.0.0.1:8073/api/v1/history", data = payload, timeout = 15).json()["data"]
        for i in range(len(response)):
            if i > 0:
                delta = response[i]["ts"] - response[i - 1]["ts"]
                if delta > 1100:
                    samples_to_fill = int(delta / (1000 / sampling_rate))
                    average_value = sum(response[i - 1]["ehz"]) / len(response[i - 1]["ehz"])
                    data["BHZ"].extend([average_value] * samples_to_fill)
            data["BHZ"].extend(response[i]["ehz"])
        start_time = current_end_time
    return data

def update(frame):
    try:
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        now_utc = datetime.datetime.utcnow().replace(tzinfo = pytz.utc)
        start_time_utc = now_utc - datetime.timedelta(hours = hours)
        start_time_utc = start_time_utc.replace(minute = 0, second = 0, microsecond = 0)
        end_time_utc = now_utc
        data = get_data(int(start_time_utc.timestamp() * 1000) + ts_offset, int(end_time_utc.timestamp() * 1000) + ts_offset)
        bhz_data = make_trace("BHZ", sampling_rate, data["BHZ"], start_time_utc.timestamp())
        bhz_data.filter("bandpass", freqmin = 0.1, freqmax = 10.0, zerophase = True)
        bhz_data.plot(ax = ax, fig = fig, type = "dayplot", title = "", color = ["k", "r", "b", "g"],
                    tick_format = "%H:%M", interval = show_interval,
                    vertical_scaling_range = scaling_range,
                    one_tick_per_line = True,
                    show_y_UTC_label = False,
                    starttime = obspy.UTCDateTime(start_time_utc),
                    endtime = obspy.UTCDateTime(end_time_utc),
                    subplots_adjust_left = 0.05, subplots_adjust_right = 1, subplots_adjust_top = 1, subplots_adjust_bottom = 0)
    except Exception as e:
        print(f"Error plotting data: {e}")

if __name__ == "__main__":
    fig, ax = matplotlib.pyplot.subplots(num = "Observer Helicorder", figsize = (9.5, 3.8))
    ani = matplotlib.animation.FuncAnimation(fig, update, interval = update_interval * 1000, cache_frame_data = False)
    matplotlib.pyplot.show()
    os._exit(0)