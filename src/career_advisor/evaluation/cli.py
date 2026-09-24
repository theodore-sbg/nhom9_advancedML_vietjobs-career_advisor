"""Tiện ích nhỏ cho các công cụ gán nhãn chạy trong terminal."""

from __future__ import annotations

import sys


def read_key() -> str:
    """Đọc một phím, không cần Enter. Khi đầu vào không phải terminal (test, pipe) thì đọc một dòng."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()[:1].lower()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
