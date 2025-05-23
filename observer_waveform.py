# -*- coding: utf-8 -*-
import os
import time
import obspy
import numpy
import socket
import struct
import threading
import matplotlib.pyplot
import matplotlib.animation

station_tcpaddr = "192.168.0.11"  # Observer TCP Forwarder address
station_tcpport = 30000  # Observer TCP Forwarder port

time_span = 120  # Time span in seconds
refresh_time = 1000  # Refresh time in milliseconds
window_size = 2  # Spectrogram window size in seconds
overlap_percent = 86  # Spectrogram overlap in percent
spectrogram_power_range = [20, 160]  # Spectrogram power range in dB

fig, axs = matplotlib.pyplot.subplots(6, 1, num="Observer Waveform", figsize=(9.6, 7.0))
matplotlib.pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)


def get_checksum(message: str) -> int:
    fields = message.split(",")
    if len(fields) < 8:
        raise ValueError("message fields length is less than 8")
    data_arr = [int(field) for field in fields[7:-1]]
    checksum = 0
    for data in data_arr:
        bytes_data = struct.pack("<i", data)
        for byte in bytes_data:
            checksum ^= byte
    return checksum


def compare_checksum(message: str):
    checksum_index = message.find("*")
    if checksum_index == -1:
        raise ValueError("checksum not found in message")
    msg_checksum = int(message[checksum_index + 1 : checksum_index + 3], 16)
    calc_checksum = get_checksum(message)
    return msg_checksum == calc_checksum


def resample_trace(trace, target_sampling_rate):
    if trace.stats.sampling_rate != target_sampling_rate:
        trace.interpolate(target_sampling_rate)
    return trace


def make_trace(net, stn, loc, channel, sps, counts_list, timestamp):
    trace = obspy.core.Trace(
        data=numpy.ma.MaskedArray(counts_list, dtype=numpy.float64)
    )
    trace.stats.network = net
    trace.stats.station = stn
    trace.stats.location = loc
    trace.stats.channel = channel
    trace.stats.sampling_rate = sps
    trace.stats.starttime = obspy.UTCDateTime(timestamp)
    return trace


def get_data(host, port):
    global bhe_data, bhn_data, bhz_data, channel_code
    buffer = ""
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((host, port))
            print(f"Connected to {host}:{port}")
            while True:
                recv_data = client_socket.recv(16384)
                if not recv_data:
                    print("No data received, connection lost.")
                    break
                buffer += recv_data.decode("utf-8")
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    if not line.strip():
                        continue
                    try:
                        if compare_checksum(line):
                            # 在确认 checksum 通过后，替换原有的字段解析部分：
                            msg = line.split("*")[0].rstrip(",")  # 去掉末尾所有逗号
                            fields = msg.split(",")
                            # ➔ ['$n', network, station, location, channel, timestamp_ms, sample_rate, sample1, …]
                            index = int(fields[0][1:])  # 序号
                            network_code = fields[1]  # 网络代码
                            station_code = fields[2]  # 观测台
                            location_code = fields[3]  # 位置代码
                            channel_code = fields[4]  # 通道，如 EHZ
                            timestamp = int(fields[5]) / 1000  # 毫秒转秒
                            sample_rate = int(fields[6])
                            samples = list(map(int, fields[7:]))
                            # 只处理轴向为 E/N/Z 的通道
                            if index in [1, 2, 3]:
                                if channel_code[2] == "E":
                                    bhe_data = make_trace(
                                        network_code,
                                        station_code,
                                        location_code,
                                        channel_code,
                                        sample_rate,
                                        samples,
                                        timestamp,
                                    )
                                elif channel_code[2] == "N":
                                    bhn_data = make_trace(
                                        network_code,
                                        station_code,
                                        location_code,
                                        channel_code,
                                        sample_rate,
                                        samples,
                                        timestamp,
                                    )
                                elif channel_code[2] == "Z":
                                    bhz_data = make_trace(
                                        network_code,
                                        station_code,
                                        location_code,
                                        channel_code,
                                        sample_rate,
                                        samples,
                                        timestamp,
                                    )
                    except Exception as ex:
                        print(f"Error processing line: {ex}")
        except Exception as e:
            print(f"Error: {e}. Reconnecting...")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            time.sleep(1)


def update(frame):
    try:
        # Resample new data to match the stream sampling rate
        bhe_resampled = resample_trace(bhe_data, bhe_stream.stats.sampling_rate)
        bhn_resampled = resample_trace(bhn_data, bhn_stream.stats.sampling_rate)
        bhz_resampled = resample_trace(bhz_data, bhz_stream.stats.sampling_rate)

        # Update streams with fixed length
        for stream, new_data in zip(
            [bhe_stream, bhn_stream, bhz_stream],
            [bhe_resampled, bhn_resampled, bhz_resampled],
        ):
            new_samples = int(new_data.stats.npts)
            stream_length = int(stream.stats.sampling_rate * time_span)

            if len(stream.data) >= stream_length:
                stream.data = numpy.roll(stream.data, -new_samples)
                stream.data[-new_samples:] = new_data.data
            else:
                stream.data = numpy.concatenate((stream.data, new_data.data))
                if len(stream.data) > stream_length:
                    stream.data = stream.data[-stream_length:]

            stream.stats.starttime = stream.stats.starttime + 1.0

        # Plot data
        for i, (stream, component) in enumerate(
            zip(
                [bhe_stream, bhn_stream, bhz_stream],
                [
                    f"{channel_code[0:2]}E",
                    f"{channel_code[0:2]}N",
                    f"{channel_code[0:2]}Z",
                ],
            )
        ):
            axs[i * 2].clear()
            axs[i * 2 + 1].clear()
            times = numpy.arange(stream.stats.npts) / stream.stats.sampling_rate
            waveform_data = (
                stream.copy()
                .filter("bandpass", freqmin=0.1, freqmax=10.0, zerophase=True)
                .data
            )

            if not numpy.any(numpy.isnan(waveform_data)) and not numpy.any(
                numpy.isinf(waveform_data)
            ):
                axs[i * 2].plot(times, waveform_data, label=component, color="blue")
                axs[i * 2].legend(loc="upper left")
                axs[i * 2].xaxis.set_visible(False)
                axs[i * 2].yaxis.set_visible(False)
                axs[i * 2].set_xlim([times[0], times[-1]])
                axs[i * 2].set_ylim(
                    [numpy.min(waveform_data), numpy.max(waveform_data)]
                )

            NFFT = int(stream.stats.sampling_rate * window_size)
            noverlap = int(NFFT * (overlap_percent / 100))
            spec_data = stream.copy().filter("highpass", freq=0.1, zerophase=True).data
            if not numpy.any(numpy.isnan(spec_data)) and not numpy.any(
                numpy.isinf(spec_data)
            ):
                axs[i * 2 + 1].specgram(
                    spec_data,
                    NFFT=NFFT,
                    Fs=stream.stats.sampling_rate,
                    noverlap=noverlap,
                    cmap="jet",
                    vmin=spectrogram_power_range[0],
                    vmax=spectrogram_power_range[1],
                )
                axs[i * 2 + 1].set_ylim(0, 15)
                axs[i * 2 + 1].yaxis.set_visible(False)
                axs[i * 2 + 1].xaxis.set_visible(False)
    except Exception as e:
        print(f"Error plotting data: {e}")


if __name__ == "__main__":
    thread1 = threading.Thread(target=get_data, args=(station_tcpaddr, station_tcpport))
    thread1.start()
    time.sleep(3)
    bhe_stream = bhe_data.copy()
    bhn_stream = bhn_data.copy()
    bhz_stream = bhz_data.copy()
    stream_length = int(bhe_stream.stats.sampling_rate * time_span)
    bhe_stream.data = numpy.zeros(stream_length)
    bhn_stream.data = numpy.zeros(stream_length)
    bhz_stream.data = numpy.zeros(stream_length)
    ani = matplotlib.animation.FuncAnimation(
        fig, update, interval=refresh_time, cache_frame_data=False
    )
    matplotlib.pyplot.show()
    os._exit(0)
