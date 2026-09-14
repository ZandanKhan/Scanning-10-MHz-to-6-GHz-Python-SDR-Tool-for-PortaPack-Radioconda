#!/usr/bin/env python3
"""
ZandanK RF Scanner - Professional SDR Suite
Refactored for thread safety, calibrated DSP scaling, and robust sweep control.
"""

import collections
import csv
from dataclasses import dataclass
import os
import queue
import sys
import time
from typing import Optional

from gnuradio import blocks, gr
import numpy as np
import osmosdr
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy.signal import find_peaks

# ---------------- Hardware & Processing Constants ----------------
DEFAULT_START_FREQ = 88e6
DEFAULT_STOP_FREQ = 108e6
DEFAULT_STEP_SIZE = 20e6
DEFAULT_SAMPLE_RATE = 10e6  # 10 MS/s ensures reliable streaming over USB
FFT_SIZE = 512
DEVICE_ARGS = "hackrf=0"
WATERFALL_DEPTH = 250
DEFAULT_CSV_PATH = "detections.csv"
MAX_MEMORY_RECORDS = 1000

# DSP and Display Constants
AUTO_Y_ALPHA = 0.90
AVG_TRACE_ALPHA = 0.85
HEADROOM_DB = 10.0
FLOOR_MARGIN_DB = 15.0
PEAK_PROMINENCE_DB = 14.0
MIN_PEAK_SPACING_HZ = 500e3  # 500 kHz physical channel isolation

SPEED_PRESETS = {
    "Fast": 0.15,
    "Medium": 0.30,
    "Slow": 0.55
}

BANDS = {
    "FM Broadcast": (88e6, 108e6),
    "Full Sweep (10M-6G)": (10e6, 6000e6),
    "4G LTE 700": (699e6, 801e6),
    "GSM 850": (824e6, 894e6),
    "GSM 900": (880e6, 960e6),
    "ADS-B": (1089e6, 1091e6),
    "GPS L1": (1574e6, 1576e6),
    "GSM 1800": (1710e6, 1880e6),
    "3G UMTS": (1920e6, 2170e6),
    "4G LTE 2100": (2110e6, 2170e6),
    "Wi-Fi 2.4 GHz": (2400e6, 2485e6),
    "Wi-Fi 5.8 GHz": (5725e6, 5875e6),
}


def classify_frequency(freq_hz: float) -> str:
    for label, (f_start, f_stop) in BANDS.items():
        if label.startswith("Full"):
            continue
        if f_start <= freq_hz <= f_stop:
            return label
    return "Unclassified"


@dataclass(frozen=True)
class ScanConfig:
    start_freq: float
    stop_freq: float
    step_size: float
    dwell_time: float


# ---------------- GNU Radio Flowgraph ----------------
class GrTop(gr.top_block):
    def __init__(self, samp_rate: float, dev_args: str):
        super().__init__()
        self.src = osmosdr.source(args=dev_args)
        self.src.set_sample_rate(samp_rate)
        self.src.set_center_freq(DEFAULT_START_FREQ)
        
        # Front-end configuration
        self.src.set_gain_mode(False)
        self.src.set_bb_gain(20)   # VGA
        self.src.set_if_gain(24)   # LNA
        self.src.set_gain(0)       # RF Amp

        self.stream_to_vec = blocks.stream_to_vector(gr.sizeof_gr_complex, FFT_SIZE)
        self.vec_sink = blocks.vector_sink_c(FFT_SIZE)
        self.connect(self.src, self.stream_to_vec, self.vec_sink)

    def set_center_freq(self, freq: float):
        self.src.set_center_freq(freq)

    def set_vga_gain(self, gain: int):
        self.src.set_bb_gain(gain)

    def set_lna_gain(self, gain: int):
        self.src.set_if_gain(gain)

    def set_amp(self, enable: bool):
        self.src.set_gain(14 if enable else 0)

    def get_data(self) -> Optional[np.ndarray]:
        data = self.vec_sink.data()
        if len(data) < FFT_SIZE:
            return None
        raw_slice = np.array(data[-FFT_SIZE:], dtype=np.complex64)
        self.vec_sink.reset()
        return raw_slice


# ---------------- SDR Worker Thread ----------------
class SdrWorker(QtCore.QThread):
    spectrum_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, float, float, float, int, int)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, sample_rate: float, dev_args: str = DEVICE_ARGS):
        super().__init__()
        self.sample_rate = sample_rate
        self.dev_args = dev_args
        self._cmd_queue = queue.Queue()
        self._running = False
        self._scanning = False
        self.tb: Optional[GrTop] = None

    def post_config(self, config: ScanConfig):
        self._cmd_queue.put(("CONFIG", config))

    def set_scanning(self, active: bool):
        self._cmd_queue.put(("SCAN_STATE", active))

    def set_gains(self, lna: Optional[int] = None, vga: Optional[int] = None, amp: Optional[bool] = None):
        self._cmd_queue.put(("GAINS", (lna, vga, amp)))

    def stop_worker(self):
        self._running = False
        self._cmd_queue.put(("TERMINATE", None))

    @staticmethod
    def _compute_steps(start_f: float, stop_f: float, step_sz: float) -> np.ndarray:
        if stop_f <= start_f or step_sz <= 0:
            return np.array([start_f], dtype=np.float64)
        steps = np.arange(start_f, stop_f + (step_sz * 0.5), step_sz)
        return np.clip(steps, start_f, stop_f)

    def run(self):
        try:
            self.tb = GrTop(self.sample_rate, self.dev_args)
            self.tb.start()
        except Exception as e:
            self.error_occurred.emit(str(e))
            return

        self._running = True
        window = np.hanning(FFT_SIZE).astype(np.float32)
        window_coherent_gain = float(np.sum(window))
        mid = FFT_SIZE // 2

        cfg = ScanConfig(
            start_freq=DEFAULT_START_FREQ,
            stop_freq=DEFAULT_STOP_FREQ,
            step_size=DEFAULT_STEP_SIZE,
            dwell_time=SPEED_PRESETS["Medium"]
        )
        steps = self._compute_steps(cfg.start_freq, cfg.stop_freq, cfg.step_size)
        step_idx = 0
        last_hop_time = time.time()

        while self._running:
            # Drain non-blocking control queue
            while not self._cmd_queue.empty():
                action, payload = self._cmd_queue.get_nowait()
                if action == "CONFIG":
                    cfg = payload
                    steps = self._compute_steps(cfg.start_freq, cfg.stop_freq, cfg.step_size)
                    step_idx = 0
                    self.tb.set_center_freq(steps[step_idx])
                    last_hop_time = time.time()
                elif action == "SCAN_STATE":
                    self._scanning = payload
                    if self._scanning:
                        last_hop_time = time.time()
                elif action == "GAINS":
                    lna, vga, amp = payload
                    if lna is not None:
                        self.tb.set_lna_gain(lna)
                    if vga is not None:
                        self.tb.set_vga_gain(vga)
                    if amp is not None:
                        self.tb.set_amp(amp)
                elif action == "TERMINATE":
                    self._running = False
                    break

            if not self._running:
                break

            if not self._scanning:
                time.sleep(0.02)
                continue

            now = time.time()
            if now - last_hop_time >= cfg.dwell_time:
                step_idx = (step_idx + 1) % len(steps)
                self.tb.set_center_freq(steps[step_idx])
                last_hop_time = now

            raw = self.tb.get_data()
            if raw is None:
                time.sleep(0.002)
                continue

            # Normalized PSD in dBFS
            fft_data = np.fft.fftshift(np.fft.fft(raw * window, n=FFT_SIZE))
            psd = 20 * np.log10((np.abs(fft_data) / window_coherent_gain) + 1e-12)
            freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / self.sample_rate)) + steps[step_idx]

            # DC Notch Interpolation
            psd[mid - 2 : mid + 3] = (psd[mid - 4] + psd[mid + 4]) / 2.0

            next_idx = (step_idx + 1) % len(steps)
            self.spectrum_ready.emit(
                freqs, psd, steps[step_idx], steps[next_idx], self.sample_rate, step_idx + 1, len(steps)
            )
            time.sleep(0.01)

        if self.tb:
            self.tb.stop()
            self.tb.wait()


# ---------------- Main User Interface ----------------
class ZandanKScanner(QtWidgets.QMainWindow):
    def __init__(self, worker: SdrWorker, sample_rate: float):
        super().__init__()
        self.worker = worker
        self.sample_rate = sample_rate

        self.waterfall_buffer = collections.deque(maxlen=WATERFALL_DEPTH)
        self.records = []
        self.recent_detections = {}
        self.max_hold_psd: Optional[np.ndarray] = None
        self.avg_psd: Optional[np.ndarray] = None

        self.auto_y_enabled = True
        self.y_min_target = -100.0
        self.y_max_target = -10.0

        # Persistent Detections Log
        self.auto_csv_file = open(DEFAULT_CSV_PATH, "a", newline="", encoding="utf-8")
        self.auto_csv_writer = csv.writer(self.auto_csv_file)
        if self.auto_csv_file.tell() == 0:
            self.auto_csv_writer.writerow(["Timestamp", "Frequency_MHz", "Power_dBFS", "Band"])
            self.auto_csv_file.flush()

        self.init_ui()
        self.apply_dark_theme()

        self.worker.spectrum_ready.connect(self.on_spectrum_received)
        self.worker.error_occurred.connect(self.on_worker_error)

    def init_ui(self):
        self.setWindowTitle("ZandanK RF Scanner")
        self.resize(1480, 930)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(8)
        main_layout.addLayout(left_layout, 4)

        right_panel = self.create_hardware_panel()
        main_layout.addWidget(right_panel, 1)

        # Action Bar
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Scan")
        self.btn_start.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.btn_start.clicked.connect(self.start_scan)

        self.btn_stop = QtWidgets.QPushButton("Stop Scan")
        self.btn_stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_scan)

        self.btn_export = QtWidgets.QPushButton("Export CSV")
        self.btn_export.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.btn_export.clicked.connect(self.export_csv_dialog)

        self.btn_reset_hold = QtWidgets.QPushButton("Clear Traces")
        self.btn_reset_hold.clicked.connect(self.reset_traces)

        lbl_speed = QtWidgets.QLabel("Speed:")
        self.combo_speed = QtWidgets.QComboBox()
        self.combo_speed.addItems(list(SPEED_PRESETS.keys()))
        self.combo_speed.setCurrentText("Medium")
        self.combo_speed.currentTextChanged.connect(self.on_config_changed)

        lbl_band = QtWidgets.QLabel("Band:")
        self.combo_band = QtWidgets.QComboBox()
        for band in BANDS:
            self.combo_band.addItem(band)
        self.combo_band.currentTextChanged.connect(self.on_config_changed)

        self.status_badge = QtWidgets.QLabel("IDLE")
        self.status_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.status_badge.setFixedSize(90, 28)
        self.status_badge.setStyleSheet("background-color: #450a0a; color: #f87171; font-weight: bold; border-radius: 4px;")

        top_bar.addWidget(self.btn_start)
        top_bar.addWidget(self.btn_stop)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(self.btn_reset_hold)
        top_bar.addSpacing(10)
        top_bar.addWidget(lbl_speed)
        top_bar.addWidget(self.combo_speed)
        top_bar.addWidget(lbl_band)
        top_bar.addWidget(self.combo_band)
        top_bar.addStretch()
        top_bar.addWidget(self.status_badge)
        left_layout.addLayout(top_bar)

        # Telemetry HUD
        hud_panel = QtWidgets.QFrame()
        hud_panel.setStyleSheet("background-color: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 6px;")
        hud_layout = QtWidgets.QHBoxLayout(hud_panel)
        hud_layout.setContentsMargins(10, 4, 10, 4)

        v_curr = QtWidgets.QVBoxLayout()
        v_curr.addWidget(QtWidgets.QLabel("TUNED FREQUENCY", styleSheet="color: #64748b; font-size: 9px; font-weight: bold;"))
        self.lbl_curr_freq = QtWidgets.QLabel("0.00 MHz")
        self.lbl_curr_freq.setStyleSheet("color: #38bdf8; font-size: 18px; font-weight: bold; font-family: 'Consolas', monospace;")
        v_curr.addWidget(self.lbl_curr_freq)

        v_next = QtWidgets.QVBoxLayout()
        v_next.addWidget(QtWidgets.QLabel("NEXT TARGET / HOP", styleSheet="color: #64748b; font-size: 9px; font-weight: bold;"))
        self.lbl_next_freq = QtWidgets.QLabel("0.00 MHz (---)")
        self.lbl_next_freq.setStyleSheet("color: #f59e0b; font-size: 13px; font-weight: bold; font-family: 'Consolas', monospace;")
        v_next.addWidget(self.lbl_next_freq)

        v_cur = QtWidgets.QVBoxLayout()
        v_cur.addWidget(QtWidgets.QLabel("CURSOR HUD", styleSheet="color: #64748b; font-size: 9px; font-weight: bold;"))
        self.lbl_cursor_val = QtWidgets.QLabel("--- MHz | --- dBFS")
        self.lbl_cursor_val.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold; font-family: 'Consolas', monospace;")
        v_cur.addWidget(self.lbl_cursor_val)

        hud_layout.addLayout(v_curr, 2)
        hud_layout.addLayout(v_next, 3)
        hud_layout.addLayout(v_cur, 2)
        left_layout.addWidget(hud_panel)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #0f172a; border: 1px solid #1e293b; border-radius: 4px; }
            QProgressBar::chunk { background-color: #38bdf8; border-radius: 3px; }
        """)
        left_layout.addWidget(self.progress_bar)

        self.tabs = QtWidgets.QTabWidget()
        left_layout.addWidget(self.tabs)

        self.tab_spectrum = QtWidgets.QWidget()
        self.tab_table = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_spectrum, "Live Spectrum & Waterfall")
        self.tabs.addTab(self.tab_table, "Detections Log")

        self.setup_spectrum_view()
        self.setup_table_view()

    def create_hardware_panel(self):
        panel = QtWidgets.QFrame()
        panel.setStyleSheet("background-color: #0b1329; border: 1px solid #1e293b; border-radius: 6px;")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl_head = QtWidgets.QLabel("FRONT-END GAIN")
        lbl_head.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        layout.addWidget(lbl_head)

        self.chk_amp = QtWidgets.QCheckBox("RF Amplifier (+14 dB)")
        self.chk_amp.toggled.connect(lambda val: self.worker.set_gains(amp=val))
        layout.addWidget(self.chk_amp)

        row_lna = QtWidgets.QHBoxLayout()
        row_lna.addWidget(QtWidgets.QLabel("LNA (IF):", styleSheet="color: #94a3b8;"))
        self.lbl_lna_val = QtWidgets.QLabel("24 dB")
        row_lna.addWidget(self.lbl_lna_val, alignment=QtCore.Qt.AlignRight)
        layout.addLayout(row_lna)

        self.slider_lna = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_lna.setRange(0, 40)
        self.slider_lna.setSingleStep(8)
        self.slider_lna.setValue(24)
        self.slider_lna.valueChanged.connect(self.on_lna_changed)
        layout.addWidget(self.slider_lna)

        row_vga = QtWidgets.QHBoxLayout()
        row_vga.addWidget(QtWidgets.QLabel("VGA (BB):", styleSheet="color: #94a3b8;"))
        self.lbl_vga_val = QtWidgets.QLabel("20 dB")
        row_vga.addWidget(self.lbl_vga_val, alignment=QtCore.Qt.AlignRight)
        layout.addLayout(row_vga)

        self.slider_vga = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_vga.setRange(0, 62)
        self.slider_vga.setSingleStep(2)
        self.slider_vga.setValue(20)
        self.slider_vga.valueChanged.connect(self.on_vga_changed)
        layout.addWidget(self.slider_vga)

        layout.addSpacing(6)
        lbl_scale = QtWidgets.QLabel("SPECTRUM Y-AXIS (dBFS)")
        lbl_scale.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        layout.addWidget(lbl_scale)

        self.chk_auto_y = QtWidgets.QCheckBox("Auto Y-Scale")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.toggled.connect(self.toggle_auto_y)
        layout.addWidget(self.chk_auto_y)

        y_ctrl_box = QtWidgets.QHBoxLayout()
        vbox_max = QtWidgets.QVBoxLayout()
        vbox_max.addWidget(QtWidgets.QLabel("Max dBFS:", styleSheet="color: #94a3b8; font-size: 10px;"))
        self.spin_y_max = QtWidgets.QSpinBox()
        self.spin_y_max.setRange(-60, 20)
        self.spin_y_max.setValue(int(self.y_max_target))
        self.spin_y_max.valueChanged.connect(self.on_manual_y_changed)
        vbox_max.addWidget(self.spin_y_max)

        vbox_min = QtWidgets.QVBoxLayout()
        vbox_min.addWidget(QtWidgets.QLabel("Min dBFS:", styleSheet="color: #94a3b8; font-size: 10px;"))
        self.spin_y_min = QtWidgets.QSpinBox()
        self.spin_y_min.setRange(-150, -20)
        self.spin_y_min.setValue(int(self.y_min_target))
        self.spin_y_min.valueChanged.connect(self.on_manual_y_changed)
        vbox_min.addWidget(self.spin_y_min)

        y_ctrl_box.addLayout(vbox_max)
        y_ctrl_box.addLayout(vbox_min)
        layout.addLayout(y_ctrl_box)

        layout.addSpacing(6)
        lbl_traces = QtWidgets.QLabel("DISPLAY TRACES")
        lbl_traces.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        layout.addWidget(lbl_traces)

        self.chk_trace_real = QtWidgets.QCheckBox("Real-Time (Cyan)")
        self.chk_trace_real.setChecked(True)
        self.chk_trace_real.toggled.connect(lambda v: self.curve_real.setVisible(v))
        layout.addWidget(self.chk_trace_real)

        self.chk_trace_max = QtWidgets.QCheckBox("Max Hold (Amber)")
        self.chk_trace_max.setChecked(True)
        self.chk_trace_max.toggled.connect(lambda v: self.curve_max.setVisible(v))
        layout.addWidget(self.chk_trace_max)

        self.chk_trace_avg = QtWidgets.QCheckBox("Average (Green)")
        self.chk_trace_avg.setChecked(False)
        self.chk_trace_avg.toggled.connect(lambda v: self.curve_avg.setVisible(v))
        layout.addWidget(self.chk_trace_avg)

        layout.addStretch()
        return panel

    def setup_spectrum_view(self):
        vbox = QtWidgets.QVBoxLayout(self.tab_spectrum)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(4)

        self.win_plots = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.win_plots)

        # Upper Spectrum Plot
        self.spec_plot = self.win_plots.addPlot(row=0, col=0)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self.spec_plot.setYRange(self.y_min_target, self.y_max_target)
        self.spec_plot.setLabel('left', "Power", units='dBFS')
        self.spec_plot.setLabel('bottom', "Frequency", units='Hz')

        self.curve_max = self.spec_plot.plot(pen=pg.mkPen(color='#f59e0b', width=1.5))
        self.curve_avg = self.spec_plot.plot(pen=pg.mkPen(color='#22c55e', width=1.2))
        self.curve_avg.setVisible(False)
        self.curve_real = self.spec_plot.plot(pen=pg.mkPen(color='#38bdf8', width=1.5))

        self.peak_markers = pg.ScatterPlotItem(size=10, pen=pg.mkPen('#ef4444', width=1.5), brush=pg.mkBrush('#ef4444'))
        self.spec_plot.addItem(self.peak_markers)
        self.peak_labels = []

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#64748b', style=QtCore.Qt.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#64748b', style=QtCore.Qt.DashLine))
        self.spec_plot.addItem(self.v_line, ignoreBounds=True)
        self.spec_plot.addItem(self.h_line, ignoreBounds=True)
        self.spec_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)

        # Lower Waterfall Plot
        self.wf_plot = self.win_plots.addPlot(row=1, col=0)
        self.wf_plot.setXLink(self.spec_plot)
        self.wf_plot.setLabel('left', "History", units='Frames')
        self.wf_plot.setLabel('bottom', "Frequency", units='Hz')

        self.img_item = pg.ImageItem()
        self.wf_plot.addItem(self.img_item)

        colors = np.array([
            (11, 19, 43, 255),
            (28, 65, 114, 255),
            (0, 168, 150, 255),
            (2, 195, 154, 255),
            (244, 162, 97, 255),
            (231, 111, 81, 255)
        ], dtype=np.ubyte)
        positions = np.array([0.0, 0.25, 0.50, 0.70, 0.88, 1.0])
        self.hist_lut = pg.HistogramLUTItem()
        self.hist_lut.setImageItem(self.img_item)
        self.hist_lut.gradient.setColorMap(pg.ColorMap(positions, colors))
        self.hist_lut.setLevels(-100, -20)
        self.win_plots.addItem(self.hist_lut, row=1, col=1)

    def setup_table_view(self):
        vbox = QtWidgets.QVBoxLayout(self.tab_table)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Frequency (MHz)", "Power (dBFS)", "Band Category"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        vbox.addWidget(self.table)

    def apply_dark_theme(self):
        self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #050b18;
            color: #f1f5f9;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
        }
        QTabBar::tab {
            background: #0f172a;
            color: #94a3b8;
            padding: 8px 18px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 3px;
        }
        QTabBar::tab:selected {
            background: #1e293b;
            color: #38bdf8;
            font-weight: bold;
        }
        QPushButton {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
            color: #f8fafc;
            padding: 6px 14px;
            font-weight: 600;
        }
        QPushButton:hover { background-color: #1e293b; border-color: #334155; }
        QPushButton:disabled { background-color: #050b18; color: #475569; border-color: #0f172a; }
        QTableWidget {
            background-color: #0b1329;
            gridline-color: #1e293b;
            border: 1px solid #1e293b;
            border-radius: 4px;
        }
        QHeaderView::section {
            background-color: #050b18;
            color: #94a3b8;
            padding: 5px;
            border: 1px solid #1e293b;
            font-weight: bold;
        }
        QComboBox, QSlider, QSpinBox {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
            color: #f8fafc;
            padding: 3px;
        }
        QCheckBox { color: #cbd5e1; font-weight: 500; }
        """)

    def toggle_auto_y(self, enabled: bool):
        self.auto_y_enabled = enabled
        self.spin_y_max.setEnabled(not enabled)
        self.spin_y_min.setEnabled(not enabled)
        if not enabled:
            self.on_manual_y_changed()

    def on_manual_y_changed(self):
        y_min = self.spin_y_min.value()
        y_max = self.spin_y_max.value()
        if y_min < y_max:
            self.spec_plot.setYRange(y_min, y_max, padding=0.0)

    def on_lna_changed(self, val: int):
        self.lbl_lna_val.setText(f"{val} dB")
        self.worker.set_gains(lna=val)

    def on_vga_changed(self, val: int):
        self.lbl_vga_val.setText(f"{val} dB")
        self.worker.set_gains(vga=val)

    def on_config_changed(self):
        f_start, f_stop = BANDS[self.combo_band.currentText()]
        dwell = SPEED_PRESETS[self.combo_speed.currentText()]
        self.worker.post_config(ScanConfig(
            start_freq=f_start,
            stop_freq=f_stop,
            step_size=DEFAULT_STEP_SIZE,
            dwell_time=dwell
        ))
        self.reset_traces()

    def reset_traces(self):
        self.max_hold_psd = None
        self.avg_psd = None
        self.waterfall_buffer.clear()
        self.curve_max.clear()
        self.curve_avg.clear()
        self.peak_markers.clear()
        for lbl in self.peak_labels:
            self.spec_plot.removeItem(lbl)
        self.peak_labels.clear()

    def start_scan(self):
        self.worker.set_scanning(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_badge.setText("ACTIVE")
        self.status_badge.setStyleSheet("background-color: #064e3b; color: #34d399; font-weight: bold; border-radius: 4px;")

    def stop_scan(self):
        self.worker.set_scanning(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_badge.setText("IDLE")
        self.status_badge.setStyleSheet("background-color: #450a0a; color: #f87171; font-weight: bold; border-radius: 4px;")

    def on_mouse_moved(self, evt):
        if self.spec_plot.sceneBoundingRect().contains(evt):
            pt = self.spec_plot.vb.mapSceneToView(evt)
            self.v_line.setPos(pt.x())
            self.h_line.setPos(pt.y())
            self.lbl_cursor_val.setText(f"{pt.x()/1e6:.2f} MHz | {pt.y():.1f} dBFS")

    def on_spectrum_received(self, freqs, psd, curr_f, next_f, samp_rate, cur_step, tot_steps):
        self.lbl_curr_freq.setText(f"{curr_f / 1e6:.2f} MHz")
        self.lbl_next_freq.setText(f"{next_f / 1e6:.2f} MHz ({classify_frequency(next_f)})")
        self.progress_bar.setMaximum(tot_steps)
        self.progress_bar.setValue(cur_step)

        if self.auto_y_enabled:
            observed_max = float(np.max(psd))
            observed_min = float(np.percentile(psd, 5))
            target_max = float(np.ceil((observed_max + HEADROOM_DB) / 10.0) * 10.0)
            target_min = float(np.floor((observed_min - FLOOR_MARGIN_DB) / 10.0) * 10.0)
            
            self.y_max_target = AUTO_Y_ALPHA * self.y_max_target + (1.0 - AUTO_Y_ALPHA) * target_max
            self.y_min_target = AUTO_Y_ALPHA * self.y_min_target + (1.0 - AUTO_Y_ALPHA) * target_min
            self.spec_plot.setYRange(self.y_min_target, self.y_max_target, padding=0.0)
            
            self.spin_y_max.blockSignals(True)
            self.spin_y_min.blockSignals(True)
            self.spin_y_max.setValue(int(self.y_max_target))
            self.spin_y_min.setValue(int(self.y_min_target))
            self.spin_y_max.blockSignals(False)
            self.spin_y_min.blockSignals(False)

        if self.max_hold_psd is None or len(self.max_hold_psd) != len(psd):
            self.max_hold_psd = np.copy(psd)
            self.avg_psd = np.copy(psd)
        else:
            self.max_hold_psd = np.maximum(self.max_hold_psd, psd)
            self.avg_psd = AVG_TRACE_ALPHA * self.avg_psd + (1.0 - AVG_TRACE_ALPHA) * psd

        self.curve_real.setData(freqs, psd)
        if self.chk_trace_max.isChecked():
            self.curve_max.setData(freqs, self.max_hold_psd)
        if self.chk_trace_avg.isChecked():
            self.curve_avg.setData(freqs, self.avg_psd)

        # Prepend row so newest spectrum stays at the top of the history
        self.waterfall_buffer.appendleft(psd)
        wf_arr = np.array(self.waterfall_buffer)
        self.img_item.setImage(wf_arr, autoLevels=False)
        self.img_item.setRect(QtCore.QRectF(freqs[0], 0, freqs[-1] - freqs[0], len(self.waterfall_buffer)))

        # Dynamic Peak Tracking
        bin_spacing = self.sample_rate / FFT_SIZE
        dist_bins = max(1, int(MIN_PEAK_SPACING_HZ / bin_spacing))
        peaks, _ = find_peaks(psd, prominence=PEAK_PROMINENCE_DB, distance=dist_bins)

        for lbl in self.peak_labels:
            self.spec_plot.removeItem(lbl)
        self.peak_labels.clear()

        marker_spots = []
        now_epoch = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        for p in peaks[:5]:
            pk_f = freqs[p]
            pk_p = psd[p]
            marker_spots.append({'pos': (pk_f, pk_p), 'data': 1})

            lbl = pg.TextItem(f"{pk_f/1e6:.1f}M", color='#f87171', anchor=(0.5, 1.3))
            lbl.setPos(pk_f, pk_p)
            self.spec_plot.addItem(lbl)
            self.peak_labels.append(lbl)

            # Deduplicate detections within 200 kHz and 3 seconds
            rounded_freq_key = round(pk_f / 200e3) * 200e3
            if rounded_freq_key in self.recent_detections:
                if now_epoch - self.recent_detections[rounded_freq_key] < 3.0:
                    continue

            self.recent_detections[rounded_freq_key] = now_epoch
            band = classify_frequency(pk_f)
            record_row = (now_str, f"{pk_f/1e6:.2f}", f"{pk_p:.1f}", band)

            self.records.append(record_row)
            if len(self.records) > MAX_MEMORY_RECORDS:
                self.records.pop(0)

            self.auto_csv_writer.writerow(list(record_row))
            self.auto_csv_file.flush()

            r_idx = self.table.rowCount()
            if r_idx >= 300:
                self.table.removeRow(0)
                r_idx -= 1
            self.table.insertRow(r_idx)
            for c_idx, val in enumerate(record_row):
                self.table.setItem(r_idx, c_idx, QtWidgets.QTableWidgetItem(val))

        self.peak_markers.setData(marker_spots)

    def export_csv_dialog(self):
        if not self.records:
            QtWidgets.QMessageBox.information(self, "Export CSV", "No signal detections logged yet.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Detection Log", f"scan_export_{int(time.time())}.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Frequency_MHz", "Power_dBFS", "Band"])
                    writer.writerows(self.records)
                QtWidgets.QMessageBox.information(self, "Success", f"Exported {len(self.records)} records.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to save CSV file:\n{str(e)}")

    def on_worker_error(self, err_msg: str):
        QtWidgets.QMessageBox.critical(self, "Hardware Error", f"SDR hardware failure:\n{err_msg}")
        self.stop_scan()

    def closeEvent(self, event):
        try:
            self.worker.spectrum_ready.disconnect(self.on_spectrum_received)
        except (TypeError, RuntimeError):
            pass

        self.worker.stop_worker()
        if not self.worker.wait(2000):
            self.worker.terminate()
            self.worker.wait(500)

        if hasattr(self, 'auto_csv_file') and not self.auto_csv_file.closed:
            self.auto_csv_file.flush()
            self.auto_csv_file.close()

        event.accept()


# ---------------- Entry Point ----------------
def main():
    pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
    app = QtWidgets.QApplication(sys.argv)

    worker = SdrWorker(DEFAULT_SAMPLE_RATE)
    worker.start()

    window = ZandanKScanner(worker, DEFAULT_SAMPLE_RATE)
    window.show()

    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()