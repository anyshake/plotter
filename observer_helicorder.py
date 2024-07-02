# -*- coding: utf-8 -*-
import os
import json
import time
import obspy
import numpy
import requests
import datetime

station_net = "AS" # Station network code
station_stn = "SHAKE" # Station station code
station_loc = "00" # Station location code
station_address = "127.0.0.1:8073" # Observer address

ts_offset = 0 # Data timestamp offset
show_interval = 60 # Helicorder Y-axes minutes
sampling_rate = 100 # Data sampling rate
scaling_range = 100000 # Helicorder scaling range

def make_trace(channel, sps, counts_list, timestamp):
    trace = obspy.core.Trace(data=numpy.ma.MaskedArray(counts_list, dtype=numpy.float64))
    trace.stats.network = station_net
    trace.stats.station = station_stn
    trace.stats.location = station_loc
    trace.stats.channel = channel
    trace.stats.sampling_rate = sps
    trace.stats.starttime = obspy.UTCDateTime(timestamp)
    return trace

def get_time():
    date = datetime.datetime.now(datetime.timezone.utc)
    timestamp = datetime.datetime.timestamp(date)
    hour = date.hour
    nowdate = date.strftime("%Y-%m-%d")
    if 0 <= hour < 12:
        nowtime = nowdate + " 00:00:00"
    else:
        nowtime = nowdate + " 12:00:00"
    date_time = datetime.datetime.strptime(nowtime, "%Y-%m-%d %H:%M:%S")
    nowtime_ts = int(date_time.timestamp() * 1000)
    return nowtime_ts

def get_data(start_time, end_time, max_interval = 3600000):
    data = {"BHZ": []}
    while start_time < end_time:
        current_end_time = min(start_time + max_interval, end_time)
        payload = {"start": start_time, "end": current_end_time, "format": "json"}
        print(payload)
        response = requests.post(f"http://{station_address}/api/v1/history", data = payload, timeout = 15).json()["data"]
        for i in range(len(response)):
            data["BHZ"].extend(response[i]["ehz"])
        start_time = current_end_time
    return data

if __name__ == "__main__":
    start_time = get_time()
    end_time = int(time.time() * 1000)
    data = get_data(start_time + ts_offset, end_time + ts_offset)
    bhz_data = make_trace("BHZ", sampling_rate, data["BHZ"], start_time / 1000)
    bhz_data.filter("bandpass", freqmin = 0.1, freqmax = 10.0, zerophase = True)
    bhz_data.plot(type = "dayplot", title = "", color = ["k", "r", "b", "g"],
            tick_format = "%H:%M", interval = show_interval,
            vertical_scaling_range = scaling_range,
            one_tick_per_line = True,
            show_y_UTC_label = False)
    os._exit(0)
