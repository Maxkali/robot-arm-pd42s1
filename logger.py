"""
统一日志模块
- 控制台输出带颜色, 文件输出纯文本
- 分级: DEBUG / INFO / WARNING / ERROR / CRITICAL
- 硬件调试时 tail -f logs/arm.log 可实时看通信帧

使用:
    from logger import get_logger
    log = get_logger('arm')
    log.info('使能电机')
    log.error('堵转', extra={'addr': 1})
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

# 颜色 (ANSI)
_COLORS = {
    'DEBUG':    '\033[36m',   # cyan
    'INFO':     '\033[32m',   # green
    'WARNING':  '\033[33m',   # yellow
    'ERROR':    '\033[31m',   # red
    'CRITICAL': '\033[35m',   # magenta
}
_RESET = '\033[0m'


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _COLORS.get(record.levelname, '')
        record.levelname = f'{color}{record.levelname}{_RESET}'
        return super().format(record)


def get_logger(name: str = 'arm', level: int = logging.INFO) -> logging.Logger:
    """获取命名 logger, 全局共享 (单例)"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    datefmt = '%H:%M:%S'

    # 控制台 (带颜色)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_ColorFormatter(fmt, datefmt))
    logger.addHandler(sh)

    # 文件 (滚动, 5MB x 3)
    fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, f'{name}.log'),
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    return logger


# 便捷函数: 记录十六进制帧 (调试串口通信时用)
def hex_dump(data: bytes) -> str:
    return ' '.join(f'{b:02X}' for b in data)
