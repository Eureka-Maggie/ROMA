import numpy as np
from scipy.io.wavfile import write

# 参数
duration = 1.0  # 秒
sample_rate = 16000  # 采样率（Hz）
frequency = 440.0  # 正弦波频率（Hz）

# 时间轴
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# 生成正弦波
waveform = 0.5 * np.sin(2 * np.pi * frequency * t)

# 转换为16位PCM格式
waveform_int16 = np.int16(waveform * 32767)

# 保存为WAV文件
write("output.wav", sample_rate, waveform_int16)
