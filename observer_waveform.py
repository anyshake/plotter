# -*- coding: utf-8 -*-
import time
import numpy
import queue
import socket
import struct
import signal
import threading
from queue import Empty
import matplotlib.pyplot
import matplotlib.animation
from matplotlib.artist import Artist
from obspy import UTCDateTime, Stream, Trace

station_tcpaddr = "127.0.0.1"  # Observer TCP Forwarder address
station_tcpport = 30000  # Observer TCP Forwarder port

time_span = 120  # Time span in seconds
refresh_time = 1000  # Refresh time in milliseconds
window_size = 2  # Spectrogram window size in seconds
overlap_percent = 86  # Spectrogram overlap in percent
spectrogram_power_range = [20, 140]  # Spectrogram power range in dB

processing_queue = queue.Queue(maxsize=1000)
fig, axs = matplotlib.pyplot.subplots(6, 1, num="Observer Waveform", figsize=(9.6, 7.0))
matplotlib.pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

stop_event = threading.Event()
worker_thread: threading.Thread | None = None
ani = None
channel_code = "EHZ"


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


def make_trace(net, stn, loc, channel, sps, counts_list, timestamp):
    trace = Trace(data=numpy.ma.MaskedArray(counts_list, dtype=numpy.float64))
    trace.stats.network = net
    trace.stats.station = stn
    trace.stats.location = loc
    trace.stats.channel = channel
    trace.stats.sampling_rate = sps
    trace.stats.starttime = UTCDateTime(timestamp)
    return trace


def resample_trace(trace, target_sampling_rate):
    if trace.stats.sampling_rate != target_sampling_rate:
        trace.interpolate(target_sampling_rate)
    return trace


def get_data(host, port):
    global channel_code
    buffer = ""

    while not stop_event.is_set():
        client_socket = None
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(1.0)  # 短超时，方便及时退出
            client_socket.connect((host, port))
            print(f"Connected to {host}:{port}")

            while not stop_event.is_set():
                try:
                    recv_data = client_socket.recv(16384)
                except socket.timeout:
                    continue

                if not recv_data:
                    print("No data received, connection lost.")
                    break

                buffer += recv_data.decode("utf-8", errors="ignore")

                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    if not line.strip():
                        continue

                    try:
                        if compare_checksum(line):
                            msg = line.split("*")[0].rstrip(",")
                            fields = msg.split(",")

                            index = int(fields[0][1:])
                            network_code = fields[1]
                            station_code = fields[2]
                            location_code = fields[3]
                            channel_code = fields[4]
                            timestamp = int(fields[5]) / 1000
                            sample_rate = int(fields[6])
                            samples = list(map(int, fields[7:]))

                            if index in [1, 2, 3] and channel_code[-1] in [
                                "E",
                                "N",
                                "Z",
                            ]:
                                trace = make_trace(
                                    network_code,
                                    station_code,
                                    location_code,
                                    channel_code,
                                    sample_rate,
                                    samples,
                                    timestamp,
                                )
                                st_data = Stream(traces=[trace])

                                if not processing_queue.empty():
                                    try:
                                        existing_stream = processing_queue.get_nowait()
                                        st_data += existing_stream
                                    except Empty:
                                        pass

                                try:
                                    processing_queue.put_nowait(st_data)
                                except queue.Full:
                                    try:
                                        processing_queue.get_nowait()
                                    except Empty:
                                        pass
                                    try:
                                        processing_queue.put_nowait(st_data)
                                    except queue.Full:
                                        pass

                    except Exception as ex:
                        if not stop_event.is_set():
                            print(f"Error processing line: {ex}")

        except Exception as e:
            if not stop_event.is_set():
                print(f"TCP Error: {e}. Reconnecting..")

        finally:
            if client_socket is not None:
                try:
                    client_socket.close()
                except Exception:
                    pass

        # 不要一次睡太久，拆成短等待，便于及时响应退出
        for _ in range(10):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def update(frame) -> tuple[Artist, ...]:
    if stop_event.is_set():
        return ()

    try:
        st_data = processing_queue.get_nowait()

        if len(st_data) < 3:
            return ()

        components = {tr.stats.channel[-1]: tr for tr in st_data}
        bhe_resampled = resample_trace(components["E"], bhe_stream.stats.sampling_rate)
        bhn_resampled = resample_trace(components["N"], bhn_stream.stats.sampling_rate)
        bhz_resampled = resample_trace(components["Z"], bhz_stream.stats.sampling_rate)

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

                ymin = float(numpy.min(waveform_data))
                ymax = float(numpy.max(waveform_data))
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                axs[i * 2].set_ylim([ymin, ymax])

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

    except Empty:
        return ()
    except Exception as e:
        if not stop_event.is_set():
            print(f"Error plotting data: {e}")

    return ()


def shutdown(*_args):
    global ani

    if stop_event.is_set():
        return

    print("Shutting down...")
    stop_event.set()

    if ani is not None:
        try:
            ani.event_source.stop()
        except Exception:
            pass

    try:
        matplotlib.pyplot.close("all")
    except Exception:
        pass


if __name__ == "__main__":
    worker_thread = threading.Thread(
        target=get_data,
        args=(station_tcpaddr, station_tcpport),
        name="observer-reader",
        daemon=True,  # 先设成 daemon，避免极端情况下解释器卡死
    )
    worker_thread.start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    fig.canvas.mpl_connect("close_event", shutdown)

    # 初始化等待首批数据，但不要无限阻塞
    st_data = None
    while not stop_event.is_set():
        try:
            st_data = processing_queue.get(timeout=0.5)
            if len(st_data) >= 3:
                break
        except Empty:
            continue

    if st_data is None or len(st_data) < 3:
        print("No initial data received, exiting.")
    else:
        components = {tr.stats.channel[-1]: tr for tr in st_data}
        bhe_stream = components["E"].copy()
        bhn_stream = components["N"].copy()
        bhz_stream = components["Z"].copy()

        stream_length = int(bhe_stream.stats.sampling_rate * time_span)
        bhe_stream.data = numpy.zeros(stream_length)
        bhn_stream.data = numpy.zeros(stream_length)
        bhz_stream.data = numpy.zeros(stream_length)

        ani = matplotlib.animation.FuncAnimation(
            fig,
            update,
            interval=refresh_time,
            cache_frame_data=False,
            blit=False,
        )

        try:
            matplotlib.pyplot.show()
        except KeyboardInterrupt:
            shutdown()
        finally:
            shutdown()
            if worker_thread is not None and worker_thread.is_alive():
                worker_thread.join(timeout=2.0)
