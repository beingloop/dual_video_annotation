# 双流视频同步标注工具 (Dual Video Annotation Tool)

这是一个基于 PyQt5 和 mpv 的双流视频同步标注工具，支持两个视频流的同步播放、倍速控制以及音频频谱分析。

## 核心功能

- **双流同步**: 支持两个视频（如主视角和辅视角）的精确同步播放。
- **点击跳转**: 进度条支持点击任意位置快速跳转。
- **音频分析**: 实时显示音频梅尔频谱图（Mel Spectrogram），辅助识别关键动作。
- **标注管理**: 支持多类别标注，自动保存至 CSV 文件，并支持点击标注区间回跳视频。
- **倍速控制**: 支持 0.5x 到 2.0x 的播放速度调节。

## 环境要求

### 1. Python 依赖
```bash
pip install -r requirements.txt
```

### 2. libmpv 动态库 (关键)
本工具依赖 `libmpv` 进行视频渲染。在 Windows 环境下，您需要 `mpv-1.dll`。

**下载地址：**
- [SourceForge libmpv](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/)
- 或者从内部服务器下载。

**安装说明：**
下载后，请将 `mpv-1.dll` 放置在项目根目录下（即 `dual_video_annotation.py` 所在目录），或者将其路径添加到系统的 `PATH` 环境变量中。

## 目录结构要求
工具期望的文件夹结构如下：
```
Parent_Folder/
├── VideoStream1/  (主视角视频)
│   └── video1.mp4
└── VideoStream2/  (辅视角视频)
    └── video1.mp4
```
两个文件夹下的视频文件名需保持一致以便自动配对。

## 快捷键
- `Space`: 播放/暂停
- `Left`: 后退 (步长可选)
- `Right`: 前进 (步长可选)
