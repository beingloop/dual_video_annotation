import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import locale

# 这是一个更健壮的路径获取方法
if getattr(sys, 'frozen', False):
    # 如果是打包后的 exe
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller --onefile 模式，文件解压到临时目录
        base_path = sys._MEIPASS
    else:
        # PyInstaller --onedir 模式，或者其他打包方式
        base_path = os.path.dirname(sys.executable)
else:
    # 如果是脚本运行，路径是脚本所在目录
    base_path = os.path.dirname(os.path.abspath(__file__))

# 然后再加到 PATH
os.environ["PATH"] = base_path + os.pathsep + os.environ["PATH"]

# 针对 Python 3.8+ 的额外保险措施
try:
    os.add_dll_directory(base_path)
except AttributeError:
    pass

# 尝试导入 mpv
MPV_AVAILABLE = False
try:
    import mpv
    MPV_AVAILABLE = True
except ImportError:
    print("错误: 未找到 python-mpv 模块。请运行 'pip install python-mpv'")
except OSError:
    print("错误: 未找到 mpv 动态库 (mpv-1.dll)。请检查")

# 尝试导入音频分析相关库
AUDIO_ANALYSIS_AVAILABLE = False
try:
    import numpy as np
    import librosa
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"音频分析库导入失败: {e}")

import pandas as pd
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QGuiApplication, QPainter, QPen, QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
# from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from PyQt5.QtMultimediaWidgets import QVideoWidget

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
# from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from PyQt5.QtMultimediaWidgets import QVideoWidget


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"}
CATEGORIES = ["动作", "社交", "屏幕", "静息", "睡眠"]


def format_timestamp(milliseconds: int) -> str:
    """按照mss格式转换毫秒时间戳。分钟不补零，秒钟补零到两位。"""
    total_seconds = max(0, int(round(milliseconds / 1000)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}{seconds:02d}"


def format_interval(start_ms: int, end_ms: int) -> str:
    """生成mss-mss区间字符串。"""
    start = min(start_ms, end_ms)
    end = max(start_ms, end_ms)
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


class VideoLoaderThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, folder_a: Path, folder_b: Path):
        super().__init__()
        self.folder_a = folder_a
        self.folder_b = folder_b

    def run(self):
        try:
            self.progress.emit("正在扫描主视角文件夹...")
            files_a = {
                file.name: file
                for file in self.folder_a.iterdir()
                if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
            }
            
            self.progress.emit("正在扫描辅视角文件夹...")
            files_b = {
                file.name: file
                for file in self.folder_b.iterdir()
                if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
            }

            self.progress.emit("正在配对视频...")
            shared = sorted(set(files_a.keys()) & set(files_b.keys()))
            
            if not shared:
                self.error.emit("两个文件夹中没有同名视频文件，无法配对。")
                return

            video_pairs = [(name, files_a[name], files_b[name]) for name in shared]
            self.finished.emit(video_pairs)
            
        except Exception as e:
            self.error.emit(f"扫描过程中发生错误: {str(e)}")


class AudioAnalysisThread(QThread):
    finished = pyqtSignal(object) # 发送 QPixmap
    error = pyqtSignal(str)

    def __init__(self, video_path: str, width: int, height: int):
        super().__init__()
        self.video_path = video_path
        self.width = width
        self.height = height

    def run(self):
        if not AUDIO_ANALYSIS_AVAILABLE:
            return

        # 检查并配置 ffmpeg 环境
        self._ensure_ffmpeg()

        try:
            # 1. 加载音频
            # librosa.load 内部会尝试使用 soundfile (不支持 avi) 和 audioread (依赖 ffmpeg)
            # 忽略 PySoundFile failed 的警告 (因为 soundfile 不支持视频容器是预期的)
            # 忽略 librosa 的 FutureWarning (关于 audioread 后端将在未来版本移除的警告)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
                warnings.filterwarnings("ignore", category=FutureWarning)
                y, sr = librosa.load(self.video_path, sr=None)
            
            # 2. 计算梅尔频谱
            # 使用较小的 hop_length 以获得更高的时间分辨率
            hop_length = 512
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # 发送原始数据而不是图片
            self.finished.emit((S_dB, sr, hop_length))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _ensure_ffmpeg(self):
        """
        检查系统是否安装了 ffmpeg。
        如果没有，尝试导入或安装 imageio-ffmpeg，并将其路径添加到环境变量 PATH 中，
        以便 librosa (audioread) 可以找到它。
        """
        import shutil
        if shutil.which("ffmpeg"):
            return # 系统中已有 ffmpeg

        print("未检测到系统 ffmpeg，正在检查 imageio-ffmpeg...")
        try:
            import imageio_ffmpeg
        except ImportError:
            print("未找到 imageio-ffmpeg，正在尝试自动安装...")
            try:
                import subprocess
                import sys
                # 尝试自动安装，使用清华源加速
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "imageio-ffmpeg", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
                ])
                import imageio_ffmpeg
            except Exception as e:
                print(f"自动安装 imageio-ffmpeg 失败: {e}")
                return

        try:
            # 获取 ffmpeg 可执行文件路径
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            
            # imageio-ffmpeg 的可执行文件通常带有版本号，如 ffmpeg-win64-v4.2.2.exe
            # 但 audioread 只会查找 "ffmpeg" 命令
            # 因此我们需要复制一份并重命名为 ffmpeg.exe
            target_ffmpeg = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
            if not os.path.exists(target_ffmpeg):
                try:
                    shutil.copy(ffmpeg_exe, target_ffmpeg)
                    print(f"已创建 ffmpeg.exe 副本于: {target_ffmpeg}")
                except Exception as e:
                    print(f"创建 ffmpeg.exe 副本失败: {e}")

            # 将其目录添加到 PATH
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
                print(f"已临时将 ffmpeg 添加到 PATH: {ffmpeg_dir}")
        except Exception as e:
            print(f"配置 imageio-ffmpeg 失败: {e}")



class MelSpectrogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.S_dB: Optional[np.ndarray] = None
        self.sr: int = 22050
        self.hop_length: int = 512
        self.current_time_ms: int = 0
        self.loading = False
        self.setStyleSheet("background-color: #2b2b2b;")
        
        # 预计算 colormap
        try:
            import matplotlib.cm as cm
            # 获取 colormap 查找表 (256, 4) -> (256, 4) uint8
            self.cmap = cm.get_cmap('magma')
            # 使用 0-1 的浮点数生成 LUT
            self.cmap_lut = (self.cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
        except:
            self.cmap_lut = None

    def load_audio(self, video_path: str):
        if not AUDIO_ANALYSIS_AVAILABLE:
            return
            
        self.S_dB = None
        self.loading = True
        self.update()
        
        # 启动线程分析
        self.thread = AudioAnalysisThread(video_path, self.width(), self.height())
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.start()

    def on_analysis_finished(self, data):
        # data is (S_dB, sr, hop_length)
        self.S_dB, self.sr, self.hop_length = data
        self.loading = False
        self.update()

    def on_analysis_error(self, msg):
        print(f"Audio analysis error: {msg}")
        self.loading = False
        self.update()

    def set_current_time(self, time_ms: int):
        self.current_time_ms = time_ms
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        if self.loading:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "正在生成频谱图...")
            return

        if self.S_dB is not None and self.cmap_lut is not None:
            # 计算当前时间对应的帧索引
            current_frame = int((self.current_time_ms / 1000) * self.sr / self.hop_length)
            
            # 计算显示窗口（前后0.5秒）
            # 0.5秒对应的帧数
            half_window_frames = int(0.5 * self.sr / self.hop_length)
            
            start_frame = current_frame - half_window_frames
            end_frame = current_frame + half_window_frames
            
            # 处理边界情况，进行填充
            n_mels, n_frames = self.S_dB.shape
            
            # 提取切片
            # 如果超出范围，需要填充
            pad_left = 0
            pad_right = 0
            
            real_start = start_frame
            real_end = end_frame
            
            if start_frame < 0:
                pad_left = -start_frame
                real_start = 0
            
            if end_frame > n_frames:
                pad_right = end_frame - n_frames
                real_end = n_frames
                
            if real_start < real_end:
                slice_data = self.S_dB[:, real_start:real_end]
            else:
                slice_data = np.zeros((n_mels, 0))

            # 归一化到 0-255
            # 假设 S_dB 范围大概在 -80 到 0
            min_db = -80.0
            max_db = 0.0
            
            norm_data = (slice_data - min_db) / (max_db - min_db)
            norm_data = np.clip(norm_data, 0, 1)
            img_data = (norm_data * 255).astype(np.uint8)
            
            # 应用 colormap
            # img_data shape: (n_mels, width)
            # mapped shape: (n_mels, width, 4)
            mapped = self.cmap_lut[img_data]
            
            # 如果需要填充
            if pad_left > 0:
                left_pad = np.zeros((n_mels, pad_left, 4), dtype=np.uint8)
                # 填充黑色或深色
                left_pad[:] = self.cmap_lut[0] 
                mapped = np.hstack((left_pad, mapped))
                
            if pad_right > 0:
                right_pad = np.zeros((n_mels, pad_right, 4), dtype=np.uint8)
                right_pad[:] = self.cmap_lut[0]
                mapped = np.hstack((mapped, right_pad))
            
            # 转换为 QImage
            # mapped 是 (height, width, 4) RGBA
            # QImage 需要 contiguous buffer
            mapped = np.ascontiguousarray(mapped)
            h, w, c = mapped.shape
            image = QImage(mapped.data, w, h, w * 4, QImage.Format_RGBA8888)
            
            # 垂直翻转（因为频谱图低频在下，矩阵索引0在上）
            # librosa specshow 默认 origin='lower'，但矩阵是 (freq, time)
            # 矩阵第0行是低频。如果不翻转，第0行会画在上面。
            # 所以我们需要镜像翻转
            image = image.mirrored(False, True)
            
            # 绘制
            painter.drawImage(self.rect(), image)

        # 绘制中心线
        center_x = self.width() // 2
        painter.setPen(QPen(QColor("#ff0000"), 2))
        painter.drawLine(center_x, 0, center_x, self.height())
        
        if not AUDIO_ANALYSIS_AVAILABLE:
             painter.setPen(Qt.gray)
             painter.drawText(self.rect(), Qt.AlignCenter, "未安装 librosa/matplotlib，无法显示频谱")


class MpvWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors)
        self.setAttribute(Qt.WA_NativeWindow)
        # 黑色背景
        self.setStyleSheet("background-color: black;")


class MpvPlayer(QObject):
    positionChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(int)
    stateChanged = pyqtSignal(int)  # 1=Playing, 2=Paused, 0=Stopped
    error = pyqtSignal(str)
    mediaStatusChanged = pyqtSignal(int) # 模拟 QMediaPlayer.MediaStatus

    # 模拟 QMediaPlayer 的常量
    StoppedState = 0
    PlayingState = 1
    PausedState = 2
    
    NoMedia = 0
    LoadingMedia = 1
    LoadedMedia = 2
    StalledMedia = 3
    BufferingMedia = 4
    BufferedMedia = 5
    EndOfMedia = 6
    InvalidMedia = 7
    UnknownMediaStatus = 8

    def __init__(self, widget: MpvWidget):
        super().__init__()
        self.widget = widget
        self._position = 0
        self._duration = 0
        self._state = MpvPlayer.StoppedState
        self._media_status = MpvPlayer.NoMedia
        self._rate = 1.0
        
        # 初始化 MPV
        # 注意：在 Windows 上需要 mpv-1.dll
        try:
            self.mpv = mpv.MPV(wid=str(int(widget.winId())), 
                               vo='gpu', 
                            #    ao='dsound',
                               # 也可以尝试 'openal' 或默认 (通常是 wasapi)
                               # 如果卡顿，可以尝试增大音频缓冲区
                               audio_buffer=0.2, 
                               keep_open='yes',
                               log_handler=self._log_handler, 
                               loglevel='warn',
                               hwdec='auto', # 开启硬解
                               )
            # 优化 Seek 体验
            self.mpv['hr-seek'] = 'yes'
        except Exception as e:
            # self.error.emit(f"MPV 初始化失败: {str(e)}")
            print(f"MPV 初始化失败: {str(e)}")
            # raise e # 暂时不抛出，避免直接崩溃，但实际上无法工作
        except Exception as e:
            # self.error.emit(f"MPV 初始化失败: {str(e)}")
            print(f"MPV 初始化失败: {str(e)}")
            # raise e # 暂时不抛出，避免直接崩溃，但实际上无法工作

        if hasattr(self, 'mpv'):
            self.mpv.observe_property('time-pos', self._on_time_pos)
            self.mpv.observe_property('duration', self._on_duration)
            self.mpv.observe_property('pause', self._on_pause)
            self.mpv.observe_property('core-idle', self._on_idle)
            self.mpv.observe_property('eof-reached', self._on_eof)

    def _log_handler(self, level, prefix, text):
        # print(f"MPV [{level}] {prefix}: {text}")
        pass

    def _on_time_pos(self, name, value):
        if value is not None:
            ms = int(value * 1000)
            self._position = ms
            self.positionChanged.emit(ms)

    def _on_duration(self, name, value):
        if value is not None:
            ms = int(value * 1000)
            self._duration = ms
            self.durationChanged.emit(ms)
            self._media_status = MpvPlayer.LoadedMedia
            self.mediaStatusChanged.emit(MpvPlayer.LoadedMedia)

    def _on_pause(self, name, value):
        new_state = MpvPlayer.PausedState if value else MpvPlayer.PlayingState
        if self._state != new_state:
            self._state = new_state
            self.stateChanged.emit(new_state)

    def _on_idle(self, name, value):
        if value:
            self._state = MpvPlayer.StoppedState
            self.stateChanged.emit(MpvPlayer.StoppedState)

    def _on_eof(self, name, value):
        if value:
            self._media_status = MpvPlayer.EndOfMedia
            self.mediaStatusChanged.emit(MpvPlayer.EndOfMedia)

    def setMedia(self, path_str: str):
        if hasattr(self, 'mpv'):
            try:
                self.mpv.loadfile(path_str)
                # 确保加载后暂停
                self.mpv.pause = True
                self._media_status = MpvPlayer.LoadingMedia
                self.mediaStatusChanged.emit(MpvPlayer.LoadingMedia)
            except Exception as e:
                print(f"Error loading file: {e}")
                self.error.emit(str(e))

    def shutdown(self):
        if hasattr(self, 'mpv'):
            self.mpv.terminate()

    def play(self):
        if hasattr(self, 'mpv'):
            self.mpv.pause = False
            # 手动更新状态以确保 UI 同步，防止 mpv 属性回调延迟或丢失导致的状态不一致
            if self._state != MpvPlayer.PlayingState:
                self._state = MpvPlayer.PlayingState
                self.stateChanged.emit(MpvPlayer.PlayingState)

    def pause(self):
        if hasattr(self, 'mpv'):
            self.mpv.pause = True
            if self._state != MpvPlayer.PausedState:
                self._state = MpvPlayer.PausedState
                self.stateChanged.emit(MpvPlayer.PausedState)

    def stop(self):
        if hasattr(self, 'mpv'):
            self.mpv.stop()
            if self._state != MpvPlayer.StoppedState:
                self._state = MpvPlayer.StoppedState
                self.stateChanged.emit(MpvPlayer.StoppedState)

    def setPosition(self, position_ms, fast=False):
        if hasattr(self, 'mpv'):
            try:
                # absolute seek
                # fast=True 使用 keyframe seek，速度快但不精确，适合拖动预览
                # fast=False 使用 exact seek，精确但稍慢
                precision = "keyframes" if fast else "exact"
                self.mpv.seek(position_ms / 1000.0, reference="absolute", precision=precision)
            except Exception as e:
                # 忽略 seek 错误，通常发生在视频未加载完成或正在切换时
                # print(f"Seek error: {e}")
                pass

    def position(self):
        return self._position

    def duration(self):
        return self._duration

    def setPlaybackRate(self, rate):
        if hasattr(self, 'mpv'):
            # 避免重复设置相同的速度，减少音频干扰
            if abs(self._rate - rate) > 0.001:
                self.mpv.speed = rate
                self._rate = rate

    def setVolume(self, volume):
        if hasattr(self, 'mpv'):
            self.mpv.volume = volume

    def setMuted(self, muted):
        if hasattr(self, 'mpv'):
            self.mpv.mute = muted
        
    def state(self):
        return self._state
        
    def mediaStatus(self):
        return self._media_status
        
    def errorString(self):
        return "MPV Error"


class AnnotationApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("双流视频同步标注工具")
        self.scale_factor = self._calculate_scale_factor()
        self.resize(self._scaled(1280), self._scaled(800))

        self.csv_path = Path("annotations.csv")
        self.csv_lock_file = None

        # 启动时检查文件占用
        if not self._try_lock_csv(show_error=True):
            sys.exit(1)
        # 锁定成功说明没被占用，先释放以便 pandas 读取
        self._unlock_csv()
        
        # 更新 CSV 列结构：Index, Path, Filename, 以及每个类别的标注和备注
        self.csv_columns = ["Index", "Path", "Filename"]
        for cat in CATEGORIES:
            self.csv_columns.extend([cat, f"{cat}_备注"])
            
        self.csv_df = self._load_or_init_dataframe()

        # 读取完毕，重新锁定文件，防止运行期间被其他程序修改
        if not self._try_lock_csv(show_error=True):
            sys.exit(1)

        self.folder_parent: Optional[Path] = None
        self.folder_a: Optional[Path] = None
        self.folder_b: Optional[Path] = None
        self.video_pairs: List[tuple[str, Path, Path]] = []
        self.current_index: int = -1
        self.current_filename: Optional[str] = None

        self.annotations: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}
        self.remarks: Dict[str, str] = {cat: "" for cat in CATEGORIES}
        self.active_starts: Dict[str, Optional[int]] = {cat: None for cat in CATEGORIES}
        self.annotation_widgets: Dict[str, Dict[str, QWidget]] = {}

        self.slider_pressed: bool = False
        self.was_playing: bool = False
        self._suppress_combo_signal: bool = False
        self.loader_thread: Optional[VideoLoaderThread] = None
        self.last_jump_interval: Optional[str] = None  # 记录上次跳转的区间，避免重复跳转
        self.current_audio_source: str = 'B' # 当前音频来源 'A' 或 'B'，默认 'B' (辅视角)

        self._build_ui()
        self._build_players()

        self.statusBar().showMessage("请选择包含 VideoStream1 和 VideoStream2 的父文件夹以开始标注")

    def _try_lock_csv(self, show_error=False) -> bool:
        """尝试锁定 CSV 文件。如果成功返回 True，否则返回 False。"""
        try:
            # 使用 'a' 模式打开，如果文件不存在会创建
            # 在 Windows 上，保持这个句柄打开通常会阻止其他程序以写模式打开
            self.csv_lock_file = open(self.csv_path, 'a')
            return True
        except PermissionError:
            if show_error:
                QMessageBox.critical(None, "文件被占用", 
                                     f"无法独占访问 {self.csv_path.name}。\n"
                                     "文件可能已被 Excel 或其他程序打开。\n"
                                     "请关闭该文件后重试。")
            return False
        except Exception as e:
            if show_error:
                QMessageBox.critical(None, "错误", f"锁定文件失败: {e}")
            return False

    def _unlock_csv(self):
        """释放 CSV 文件锁"""
        if self.csv_lock_file:
            try:
                self.csv_lock_file.close()
            except Exception:
                pass
            self.csv_lock_file = None

    def _load_or_init_dataframe(self) -> pd.DataFrame:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
                missing_cols = [col for col in self.csv_columns if col not in df.columns]
                for col in missing_cols:
                    df[col] = ""
                return df[self.csv_columns]
            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                print(f"读取 CSV 出错: {e}")
        
        return pd.DataFrame(columns=self.csv_columns)

    def _calculate_scale_factor(self) -> float:
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return 1.0
        geometry = screen.availableGeometry()
        width_ratio = geometry.width() / 1280
        height_ratio = geometry.height() / 800
        ratio = min(width_ratio, height_ratio)
        # 调整缩放系数，使其更保守，避免界面过大
        return max(0.6, min(1.5, ratio * 0.9))

    def _scaled(self, value: int) -> int:
        return max(1, int(round(value * self.scale_factor)))

    def _scaled_font(self, point_size: int) -> int:
        # 字体缩放也稍微保守一点
        return max(8, int(round(point_size * self.scale_factor * 0.9)))

    def _build_ui(self) -> None:
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(self._scaled(8))  # 减小间距
        main_layout.setContentsMargins(
            self._scaled(10),
            self._scaled(10),
            self._scaled(10),
            self._scaled(10),
        )

        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(self._scaled(8))

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("请输入包含 VideoStream1/2 的父文件夹路径")
        
        self.confirm_btn = QPushButton("确定")
        self.confirm_btn.clicked.connect(self.on_confirm_path)

        folder_layout.addWidget(self.path_input, stretch=1)
        folder_layout.addWidget(self.confirm_btn)

        navigation_layout = QHBoxLayout()
        navigation_layout.setSpacing(self._scaled(8))

        self.prev_btn = QPushButton("上一个视频")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.show_previous_video)

        self.pair_combo = QComboBox()
        self.pair_combo.setEnabled(False)
        self.pair_combo.currentIndexChanged.connect(self._on_combo_changed)
        self.pair_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # 让下拉框尽可能宽

        self.next_btn = QPushButton("下一个视频")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.show_next_video)

        navigation_layout.addWidget(self.prev_btn)
        navigation_layout.addWidget(self.pair_combo, stretch=1)
        navigation_layout.addWidget(self.next_btn)

        video_layout = QHBoxLayout()
        video_layout.setSpacing(self._scaled(8))

        self.video_widget_a = MpvWidget()
        # 移除固定最小尺寸，改用较小的最小值，允许缩小
        self.video_widget_a.setMinimumSize(self._scaled(320), self._scaled(180))
        self.video_widget_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.video_widget_b = MpvWidget()
        self.video_widget_b.setMinimumSize(self._scaled(320), self._scaled(180))
        self.video_widget_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_layout.addWidget(self.video_widget_a, stretch=1)
        video_layout.addWidget(self.video_widget_b, stretch=1)

        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(self._scaled(8))

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.sliderMoved.connect(self.on_slider_moved)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(
            f"color: #333333; font-weight: bold; font-size: {self._scaled_font(10)}pt;"
        )

        progress_layout.addWidget(self.position_slider, stretch=1)
        progress_layout.addWidget(self.time_label)

        # 梅尔频谱显示区域
        self.spectrogram_widget = MelSpectrogramWidget()
        self.spectrogram_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spectrogram_widget.setFixedHeight(self._scaled(100))

        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(self._scaled(10))

        self.play_button = QPushButton("播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)

        # 快进快退步长选择
        seek_label = QLabel("快进/退:")
        self.seek_step_combo = QComboBox()
        self.seek_step_combo.addItems(["1s", "2s", "5s", "10s", "30s"])
        self.seek_step_combo.setCurrentText("5s")
        self.seek_step_combo.setFixedWidth(self._scaled(60))

        speed_label = QLabel("倍速:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setEnabled(False)
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)

        self.audio_source_btn = QPushButton("音频源: 辅视角")
        self.audio_source_btn.setCheckable(True)
        self.audio_source_btn.setChecked(True) # 默认 B (辅视角)
        self.audio_source_btn.clicked.connect(self.toggle_audio_source)
        self.audio_source_btn.setEnabled(False)

        playback_layout.addWidget(self.play_button)
        playback_layout.addSpacing(self._scaled(16))
        playback_layout.addWidget(seek_label)
        playback_layout.addWidget(self.seek_step_combo)
        playback_layout.addSpacing(self._scaled(16))
        playback_layout.addWidget(speed_label)
        playback_layout.addWidget(self.speed_combo)
        playback_layout.addSpacing(self._scaled(16))
        playback_layout.addWidget(self.audio_source_btn)
        playback_layout.addStretch()

        annotations_frame = QFrame()
        annotations_frame.setFrameShape(QFrame.StyledPanel)
        annotations_layout = QVBoxLayout(annotations_frame)
        annotations_layout.setSpacing(self._scaled(6)) # 减小标注行间距
        annotations_layout.setContentsMargins(
            self._scaled(8),
            self._scaled(8),
            self._scaled(8),
            self._scaled(8),
        )

        for category in CATEGORIES:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(self._scaled(6))

            label = QLabel(category)
            label.setFixedWidth(self._scaled(40)) # 稍微减小宽度

            toggle_btn = QToolButton()
            toggle_btn.setText("开始/结束")
            toggle_btn.setCheckable(True)
            toggle_btn.setMinimumWidth(self._scaled(80)) # 减小宽度
            toggle_btn.toggled.connect(lambda checked, cat=category: self.on_toggle_annotation(cat, checked))

            full_btn = QPushButton("全段")
            full_btn.setFixedWidth(self._scaled(50)) # 减小宽度
            full_btn.clicked.connect(lambda _, cat=category: self.on_full_annotation(cat))

            display = QLineEdit()
            display.setPlaceholderText("尚未标注")
            # 允许编辑，并连接信号以保存更改
            display.setReadOnly(False)
            display.editingFinished.connect(lambda cat=category: self.on_manual_edit(cat))
            # 安装事件过滤器以监听点击和光标移动
            display.installEventFilter(self)

            remark = QLineEdit()
            remark.setPlaceholderText("备注")
            remark.editingFinished.connect(lambda cat=category: self.on_remark_edit(cat))

            row_layout.addWidget(label)
            row_layout.addWidget(toggle_btn)
            row_layout.addWidget(full_btn)
            row_layout.addWidget(display, stretch=2)
            row_layout.addWidget(remark, stretch=1)

            annotations_layout.addLayout(row_layout)
            self.annotation_widgets[category] = {
                "toggle": toggle_btn,
                "full": full_btn,
                "display": display,
                "remark": remark,
            }

        main_layout.addLayout(folder_layout)
        main_layout.addLayout(navigation_layout)
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addWidget(self.spectrogram_widget) # 添加频谱图
        main_layout.addLayout(progress_layout)
        main_layout.addLayout(playback_layout)
        main_layout.addWidget(annotations_frame)

        self.setCentralWidget(central_widget)

        style_sheet = f"""
        QWidget {{ font-family: 'Microsoft YaHei'; font-size: {self._scaled_font(10)}pt; }}
        QPushButton, QToolButton {{ 
            padding: {self._scaled(4)}px {self._scaled(8)}px; 
            border-radius: {self._scaled(4)}px; 
            background-color: #e0e0e0; 
            border: 1px solid #c0c0c0;
        }}
        QPushButton:enabled:hover, QToolButton:enabled:hover {{ 
            background-color: #d0d0d0; 
            border: 1px solid #a0a0a0;
        }}
        QPushButton:pressed, QToolButton:pressed {{ 
            background-color: #c0c0c0; 
        }}
        QPushButton:disabled, QToolButton:disabled {{ 
            background-color: #f5f5f5; 
            color: #a0a0a0;
            border: 1px solid #e0e0e0;
        }}
        QLineEdit {{ padding: {self._scaled(3)}px {self._scaled(5)}px; background-color: #fafafa; border: 1px solid #dddddd; }}
        QComboBox {{ padding: {self._scaled(3)}px {self._scaled(5)}px; }}
        
        /* 增强进度条样式 */
        QSlider::groove:horizontal {{
            border: 1px solid #999999;
            height: {self._scaled(10)}px; 
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
            margin: 2px 0;
            border-radius: {self._scaled(5)}px;
        }}
        QSlider::handle:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: {self._scaled(18)}px;
            margin: -{self._scaled(5)}px 0; 
            border-radius: {self._scaled(3)}px;
        }}
        """
        self.setStyleSheet(style_sheet)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """监听标注输入框的点击和光标移动事件，实现点击跳转功能"""
        if event.type() == QEvent.MouseButtonRelease or event.type() == QEvent.KeyRelease:
            # 检查是否是我们的标注输入框
            for category, widgets in self.annotation_widgets.items():
                if obj == widgets["display"]:
                    self._handle_cursor_jump(widgets["display"])
                    break
        return super().eventFilter(obj, event)

    def _handle_cursor_jump(self, line_edit: QLineEdit) -> None:
        """根据光标位置跳转视频"""
        text = line_edit.text()
        cursor_pos = line_edit.cursorPosition()
        
        # 找到光标所在的区间
        current_pos = 0
        target_interval = None
        
        # 分割并保留分隔符位置信息
        parts = text.split(';')
        for part in parts:
            # 加上分号的长度（除了最后一个）
            part_len = len(part) + 1 
            if current_pos <= cursor_pos <= current_pos + part_len:
                target_interval = part.strip()
                break
            current_pos += part_len
            
        if not target_interval:
            self.last_jump_interval = None
            return

        # 如果还在同一个区间内，不重复跳转
        if target_interval == self.last_jump_interval:
            return

        # 解析开始时间
        try:
            if '-' in target_interval:
                start_str = target_interval.split('-')[0].strip()
                # 解析 mss 格式 (例如 105 -> 1分05秒 -> 65000ms)
                if len(start_str) >= 3:
                    seconds = int(start_str[-2:])
                    minutes = int(start_str[:-2])
                    total_ms = (minutes * 60 + seconds) * 1000
                    
                    # 执行跳转
                    self.player_a.setPosition(total_ms)
                    self.player_b.setPosition(total_ms)
                    self.last_jump_interval = target_interval
                    # print(f"DEBUG: Jump to {target_interval} -> {total_ms}ms")
        except ValueError:
            pass

    def _build_players(self) -> None:
        self.player_a = MpvPlayer(self.video_widget_a)
        self.player_b = MpvPlayer(self.video_widget_b)

        # 连接信号
        self.player_a.positionChanged.connect(self.on_position_changed)
        self.player_a.durationChanged.connect(self.on_duration_changed)
        self.player_a.stateChanged.connect(self.on_state_changed)
        self.player_a.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player_a.error.connect(self.on_media_error)
        self.player_b.error.connect(self.on_media_error)

    def on_confirm_path(self) -> None:
        path_str = self.path_input.text().strip()
        if not path_str:
            return

        directory = Path(path_str)
        if not directory.exists() or not directory.is_dir():
            QMessageBox.warning(self, "路径错误", "输入的路径不存在或不是文件夹。")
            return

        self.folder_parent = directory
        
        self.folder_a = self.folder_parent / "VideoStream1"
        self.folder_b = self.folder_parent / "VideoStream2"
        
        if not self.folder_a.exists() or not self.folder_b.exists():
            QMessageBox.warning(self, "文件夹结构错误", 
                                f"在所选目录下未找到 VideoStream1 或 VideoStream2 文件夹。\n"
                                f"请确保目录结构正确。\n"
                                f"当前选择: {self.folder_parent}")
            return
            
        self.start_loading_videos()

    def start_loading_videos(self) -> None:
        if not self.folder_a or not self.folder_b:
            return

        # 禁用UI防止操作
        self.confirm_btn.setEnabled(False)
        self.path_input.setEnabled(False)
        self.statusBar().showMessage("正在准备加载视频...")

        self.loader_thread = VideoLoaderThread(self.folder_a, self.folder_b)
        self.loader_thread.progress.connect(self.update_loading_status)
        self.loader_thread.finished.connect(self.on_loading_finished)
        self.loader_thread.error.connect(self.on_loading_error)
        self.loader_thread.start()

    def update_loading_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def on_loading_error(self, message: str) -> None:
        self.confirm_btn.setEnabled(True)
        self.path_input.setEnabled(True)
        self.statusBar().showMessage(f"加载失败: {message}")
        QMessageBox.warning(self, "加载错误", message)

    def on_loading_finished(self, video_pairs: List[tuple[str, Path, Path]]) -> None:
        self.confirm_btn.setEnabled(True)
        self.path_input.setEnabled(True)
        self.video_pairs = video_pairs
        
        shared_names = [p[0] for p in self.video_pairs]
        
        self._suppress_combo_signal = True
        self.pair_combo.clear()
        self.pair_combo.addItems(shared_names)
        self.pair_combo.setEnabled(True)
        self._suppress_combo_signal = False

        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.play_button.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.position_slider.setEnabled(True)
        self.audio_source_btn.setEnabled(True)

        self.load_video_at_index(0)
        self.statusBar().showMessage(f"加载完成，已配对 {len(self.video_pairs)} 个视频，当前加载 {shared_names[0]}")

    def load_video_at_index(self, index: int) -> None:
        if index < 0 or index >= len(self.video_pairs):
            return

        if self.current_filename is not None:
            self.persist_annotations()

        self.current_index = index
        name, path_a, path_b = self.video_pairs[index]
        self.current_filename = name

        self._suppress_combo_signal = True
        self.pair_combo.setCurrentIndex(index)
        self._suppress_combo_signal = False

        self.load_media_sources(path_a, path_b)
        
        # 加载音频频谱
        if AUDIO_ANALYSIS_AVAILABLE:
            audio_path = path_a if self.current_audio_source == 'A' else path_b
            self.spectrogram_widget.load_audio(str(audio_path.resolve()))
            
        self.reset_annotation_state()
        self.restore_annotations_for_current()
        self.update_navigation_buttons()
        self.statusBar().showMessage(f"已切换至 {name}", 5000)

    def load_media_sources(self, path_a: Path, path_b: Path) -> None:
        print(f"DEBUG: Loading video A: {path_a}")
        print(f"DEBUG: Loading video B: {path_b}")
        
        # self.player_a.stop()
        # self.player_b.stop()

        # 确保路径是绝对路径
        abs_path_a = path_a.resolve()
        abs_path_b = path_b.resolve()

        self.player_a.setMedia(str(abs_path_a))
        self.player_b.setMedia(str(abs_path_b))
        
        # 根据当前音频源设置静音
        if self.current_audio_source == 'A':
            self.player_a.setMuted(False)
            self.player_a.setVolume(80)
            self.player_b.setMuted(True)
        else:
            self.player_a.setMuted(True)
            self.player_b.setMuted(False)
            self.player_b.setVolume(80)

        self.player_a.setPlaybackRate(self.get_selected_speed())
        self.player_b.setPlaybackRate(self.get_selected_speed())

        self.position_slider.blockSignals(True)
        self.position_slider.setValue(0)
        self.position_slider.blockSignals(False)
        self.time_label.setText("00:00 / 00:00")

        self.play_button.setText("播放")
        
        # 尝试强制刷新第一帧
        self.player_a.pause()
        self.player_b.pause()

    def reset_annotation_state(self) -> None:
        for category in CATEGORIES:
            self.annotations[category] = []
            self.remarks[category] = ""
            self.active_starts[category] = None
            widgets = self.annotation_widgets[category]
            
            toggle_btn: QToolButton = widgets["toggle"]  # type: ignore[assignment]
            toggle_btn.blockSignals(True)
            toggle_btn.setChecked(False)
            toggle_btn.setText("开始/结束")
            toggle_btn.setStyleSheet("")
            toggle_btn.blockSignals(False)
            
            display: QLineEdit = widgets["display"]  # type: ignore[assignment]
            display.clear()
            
            remark: QLineEdit = widgets["remark"]  # type: ignore[assignment]
            remark.clear()

    def restore_annotations_for_current(self) -> None:
        if self.current_filename is None:
            return
        if self.csv_df.empty:
            return

        mask = self.csv_df["Filename"] == self.current_filename
        if not mask.any():
            return

        row = self.csv_df.loc[mask].iloc[0]
        for category in CATEGORIES:
            raw_text = row.get(category, "") or ""
            entries = [seg.strip() for seg in raw_text.split(";") if seg.strip()]
            self.annotations[category] = entries
            display: QLineEdit = self.annotation_widgets[category]["display"]  # type: ignore[assignment]
            display.setText("; ".join(entries))
            
            remark_text = row.get(f"{category}_备注", "") or ""
            self.remarks[category] = remark_text
            remark: QLineEdit = self.annotation_widgets[category]["remark"]  # type: ignore[assignment]
            remark.setText(remark_text)

    def update_navigation_buttons(self) -> None:
        total = len(self.video_pairs)
        self.prev_btn.setEnabled(total > 1 and self.current_index > 0)
        self.next_btn.setEnabled(total > 1 and self.current_index < total - 1)

    def _on_combo_changed(self, index: int) -> None:
        if self._suppress_combo_signal:
            return
        self.load_video_at_index(index)

    def show_previous_video(self) -> None:
        if self.current_index > 0:
            self.load_video_at_index(self.current_index - 1)

    def show_next_video(self) -> None:
        if self.current_index < len(self.video_pairs) - 1:
            self.load_video_at_index(self.current_index + 1)

    def toggle_audio_source(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.video_pairs):
            return

        # 切换状态
        if self.current_audio_source == 'B':
            self.current_audio_source = 'A'
            self.audio_source_btn.setText("音频源: 主视角")
            self.audio_source_btn.setChecked(False)
            
            # 切换播放器静音状态
            self.player_a.setMuted(False)
            self.player_a.setVolume(80)
            self.player_b.setMuted(True)
            
            # 重新加载频谱图
            if AUDIO_ANALYSIS_AVAILABLE:
                path = self.video_pairs[self.current_index][1] # path_a
                self.spectrogram_widget.load_audio(str(path.resolve()))
        else:
            self.current_audio_source = 'B'
            self.audio_source_btn.setText("音频源: 辅视角")
            self.audio_source_btn.setChecked(True)
            
            # 切换播放器静音状态
            self.player_a.setMuted(True)
            self.player_b.setMuted(False)
            self.player_b.setVolume(80)
            
            # 重新加载频谱图
            if AUDIO_ANALYSIS_AVAILABLE:
                path = self.video_pairs[self.current_index][2] # path_b
                self.spectrogram_widget.load_audio(str(path.resolve()))

    def toggle_playback(self) -> None:
        if self.player_a.mediaStatus() in (MpvPlayer.NoMedia, MpvPlayer.InvalidMedia):
            return
        if self.player_a.state() == MpvPlayer.PlayingState:
            self.player_a.pause()
            self.player_b.pause()
        else:
            self.player_b.setPosition(self.player_a.position())
            self.player_a.play()
            self.player_b.play()

    def on_state_changed(self, state: int) -> None:
        self.play_button.setText("暂停" if state == MpvPlayer.PlayingState else "播放")

    def on_duration_changed(self, duration: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, duration)
        self.position_slider.blockSignals(False)
        self.update_time_label(self.player_a.position(), duration)

    def on_position_changed(self, position: int) -> None:
        if not self.slider_pressed:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(position)
            self.position_slider.blockSignals(False)

        duration = max(self.player_a.duration(), 1)
        self.update_time_label(position, duration)
        
        # 更新频谱图进度
        if duration > 0 and AUDIO_ANALYSIS_AVAILABLE:
            self.spectrogram_widget.set_current_time(position)

        # 同步逻辑优化 v2：以音频源 (Player B) 为基准，调整视频源 (Player A) 的速度
        # 这样可以保证音频流畅，不会出现爆音或变调
        if self.player_a.state() == MpvPlayer.PlayingState:
            pos_b = self.player_b.position()
            diff = pos_b - position # B - A (B是基准)
            
            target_speed = self.get_selected_speed()
            
            # 确保 B 始终以目标速度播放 (音频基准)
            # 这里加个简单的检查，避免每帧都设置
            # 注意：MPV 的 speed 属性读取可能有点耗时，这里假设 B 速度正确，或者在 on_speed_changed 保证
            
            # 误差超过 2秒，或者 B 已经停止，则强制 Seek A 到 B 的位置 (或者 B 到 A? A是主控)
            # 既然 A 驱动 UI，我们还是让 B 跟随 A 的大跳跃，但微调时 A 跟随 B
            if abs(diff) > 2000 or self.player_b.state() != MpvPlayer.PlayingState:
                # 严重不同步，强制 B 同步到 A (因为 A 是 UI 主控)
                self.player_b.blockSignals(True)
                self.player_b.setPosition(position)
                self.player_b.blockSignals(False)
                self.player_b.setPlaybackRate(target_speed)
                self.player_a.setPlaybackRate(target_speed)
            
            # 误差超过 100ms，调整 A 的速度来追赶 B
            elif abs(diff) > 100:
                # 如果 B 比 A 快 (diff > 0)，A 需要加速追赶
                # 如果 B 比 A 慢 (diff < 0)，A 需要减速等待
                adjust = 1.10 if diff > 0 else 0.90
                new_speed = target_speed * adjust
                self.player_a.setPlaybackRate(new_speed)
                # B 保持原速
                self.player_b.setPlaybackRate(target_speed)
            else:
                # 误差在允许范围内，A 恢复正常速度
                self.player_a.setPlaybackRate(target_speed)
                self.player_b.setPlaybackRate(target_speed)
        else:
            # 暂停状态下，保持严格同步
            lag = abs(self.player_b.position() - position)
            if lag > 40:
                self.player_b.blockSignals(True)
                self.player_b.setPosition(position)
                self.player_b.blockSignals(False)

    def on_slider_pressed(self) -> None:
        self.slider_pressed = True
        self.was_playing = (self.player_a.state() == MpvPlayer.PlayingState)
        if self.was_playing:
            self.player_a.pause()
            self.player_b.pause()

    def on_slider_released(self) -> None:
        if not self.slider_pressed:
            return
        target = self.position_slider.value()
        self.player_a.setPosition(target, fast=False) # 释放时精确 Seek
        self.player_b.setPosition(target, fast=False)
        self.slider_pressed = False
        
        if self.was_playing:
            self.player_a.play()
            self.player_b.play()

    def on_slider_moved(self, value: int) -> None:
        # 实时预览：拖动时使用 fast seek (keyframe) 以获得高刷新率，避免重复精确 seek 和重复更新时间显示
        self.player_a.setPosition(value, fast=False)
        self.player_b.setPosition(value, fast=False)
        
        duration = max(self.player_a.duration(), 1)
        self.update_time_label(value, duration)

    def update_time_label(self, position: int, duration: int) -> None:
        current_text = self.format_time_display(position)
        duration_text = self.format_time_display(duration)
        self.time_label.setText(f"{current_text} / {duration_text}")

    @staticmethod
    def format_time_display(milliseconds: int) -> str:
        total_seconds = max(0, int(round(milliseconds / 1000)))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def on_speed_changed(self, index: int) -> None:
        speed = self.get_selected_speed()
        self.player_a.setPlaybackRate(speed)
        self.player_b.setPlaybackRate(speed)

    def get_selected_speed(self) -> float:
        text = self.speed_combo.currentText().replace("x", "")
        try:
            return float(text)
        except ValueError:
            return 1.0

    def on_toggle_annotation(self, category: str, checked: bool) -> None:
        widgets = self.annotation_widgets[category]
        toggle_btn: QToolButton = widgets["toggle"]  # type: ignore[assignment]

        if checked:
            self.active_starts[category] = self.player_a.position()
            toggle_btn.setText("结束")
            toggle_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        else:
            start_ms = self.active_starts.get(category)
            if start_ms is None:
                toggle_btn.setText("开始/结束")
                toggle_btn.setStyleSheet("")
                return
            end_ms = self.player_a.position()
            interval = format_interval(start_ms, end_ms)
            self.annotations[category].append(interval)
            self.active_starts[category] = None
            toggle_btn.setText("开始/结束")
            toggle_btn.setStyleSheet("")
            self.update_annotation_display(category)
            self.persist_annotations()

    def on_full_annotation(self, category: str) -> None:
        duration = self.player_a.duration()
        if duration <= 0:
            QMessageBox.information(self, "尚未加载", "请先加载视频后再使用全段标注。")
            return
        interval = format_interval(0, duration)
        if interval not in self.annotations[category]:
            self.annotations[category].append(interval)
            self.update_annotation_display(category)
            self.persist_annotations()

    def on_manual_edit(self, category: str) -> None:
        """处理手动编辑标注文本框的事件"""
        display: QLineEdit = self.annotation_widgets[category]["display"]  # type: ignore[assignment]
        text = display.text().strip()
        
        # 解析文本框内容回 annotations 列表
        if not text:
            self.annotations[category] = []
        else:
            # 简单的分割处理，用户可以用分号分隔多个区间
            # 这里不做严格的格式校验，允许用户自由输入备注等
            entries = [seg.strip() for seg in text.split(";") if seg.strip()]
            self.annotations[category] = entries
            
        self.persist_annotations()

    def on_remark_edit(self, category: str) -> None:
        """处理手动编辑备注文本框的事件"""
        remark_widget: QLineEdit = self.annotation_widgets[category]["remark"]  # type: ignore[assignment]
        self.remarks[category] = remark_widget.text().strip()
        self.persist_annotations()

    def update_annotation_display(self, category: str) -> None:
        display: QLineEdit = self.annotation_widgets[category]["display"]  # type: ignore[assignment]
        display.setText("; ".join(self.annotations[category]))

    def persist_annotations(self) -> None:
        if not self.current_filename:
            return
            
        path_str = ""
        if self.current_index >= 0 and self.current_index < len(self.video_pairs):
             # video_pairs is list of (name, path_a, path_b)
             path_str = str(self.video_pairs[self.current_index][1])

        row_data = {
            "Index": str(self.current_index + 1),
            "Path": path_str,
            "Filename": self.current_filename
        }
        
        for category in CATEGORIES:
            row_data[category] = "; ".join(self.annotations[category])
            row_data[f"{category}_备注"] = self.remarks[category]

        mask = self.csv_df["Filename"] == self.current_filename
        if mask.any():
            index = self.csv_df.index[mask][0]
            for key, value in row_data.items():
                self.csv_df.at[index, key] = value
        else:
            new_row = pd.DataFrame([row_data], columns=self.csv_columns)
            self.csv_df = pd.concat([self.csv_df, new_row], ignore_index=True)

        self.csv_df = self.csv_df[self.csv_columns]
        
        # 临时释放锁以允许 pandas 写入
        self._unlock_csv()
        try:
            self.csv_df.to_csv(self.csv_path, index=False)
        except PermissionError:
            QMessageBox.critical(self, "保存失败", f"无法写入 {self.csv_path.name}。\n文件可能被其他程序占用。")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入文件时发生错误:\n{e}")
        finally:
            # 重新锁定
            if not self._try_lock_csv():
                self.statusBar().showMessage("警告：无法重新锁定 CSV 文件", 5000)

    def on_media_status_changed(self, status: int) -> None:
        # 打印详细的状态变化，观察拖动时是否频繁进入 Buffering 或 Stalled 状态
        status_str = {
            MpvPlayer.UnknownMediaStatus: "Unknown",
            MpvPlayer.NoMedia: "NoMedia",
            MpvPlayer.LoadingMedia: "Loading",
            MpvPlayer.LoadedMedia: "Loaded",
            MpvPlayer.StalledMedia: "Stalled",
            MpvPlayer.BufferingMedia: "Buffering",
            MpvPlayer.BufferedMedia: "Buffered",
            MpvPlayer.EndOfMedia: "End",
            MpvPlayer.InvalidMedia: "Invalid"
        }.get(status, str(status))
        
        print(f"DEBUG: Media Status Changed: {status_str}")
        
        if status == MpvPlayer.EndOfMedia:
            self.player_a.pause()
            self.player_b.pause()
        elif status == MpvPlayer.InvalidMedia:
            self.statusBar().showMessage("媒体无效，可能是解码器缺失或文件损坏", 5000)
        elif status == MpvPlayer.LoadedMedia:
            # 视频加载成功后，尝试显示第一帧
            self.player_a.setPosition(0)
            self.player_b.setPosition(0)

    def on_media_error(self) -> None:
        error_message = self.player_a.errorString() or self.player_b.errorString()
        print(f"DEBUG: Media Error: {error_message}")
        if error_message:
            QMessageBox.critical(self, "播放错误", 
                                 f"无法播放视频。\n错误信息: {error_message}\n\n"
                                 "提示: AVI格式通常需要安装额外的解码器。\n"
                                 "请尝试安装 'LAV Filters' 或 'K-Lite Codec Pack'。")

    def keyPressEvent(self, event):
        # 如果焦点在输入框上，不处理快捷键，让输入框处理（移动光标等）
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, QLineEdit):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Left:
            self.seek_relative(-self.get_seek_step())
        elif event.key() == Qt.Key_Right:
            self.seek_relative(self.get_seek_step())
        elif event.key() == Qt.Key_Space:
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

    def get_seek_step(self) -> int:
        text = self.seek_step_combo.currentText()
        try:
            return int(text.replace('s', '')) * 1000
        except ValueError:
            return 5000

    def seek_relative(self, delta_ms: int):
        if self.player_a.mediaStatus() in (MpvPlayer.NoMedia, MpvPlayer.InvalidMedia):
            return
        
        current_pos = self.player_a.position()
        duration = self.player_a.duration()
        new_pos = max(0, min(duration, current_pos + delta_ms))
        
        self.player_a.setPosition(new_pos)
        self.player_b.setPosition(new_pos)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.persist_annotations()
        self._unlock_csv()
        if hasattr(self, 'player_a'):
            self.player_a.shutdown()
        if hasattr(self, 'player_b'):
            self.player_b.shutdown()
        super().closeEvent(event)


def main() -> None:
    # 启用高DPI缩放支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)

    if not MPV_AVAILABLE:
        QMessageBox.critical(None, "缺少组件", 
                             "未找到 mpv 动态库 (mpv-1.dll)。\n"
                             "请下载 libmpv (Windows版) 并将 .dll 放入程序运行目录或系统 PATH 中。\n"
                             "下载地址参考: https://sourceforge.net/projects/mpv-player-windows/files/libmpv/")
        sys.exit(1)

    window = AnnotationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
