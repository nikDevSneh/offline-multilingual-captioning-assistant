"""
Always-on-top, borderless caption overlay window.

Runs Tkinter's mainloop on the main thread and exposes a thread-safe
`update(text)` method the pipeline worker thread can call to push new
caption text — Tkinter widgets must only be touched from the GUI thread, so
`update()` schedules the change via `after()` instead of writing directly.
"""
from __future__ import annotations

import queue
import tkinter as tk
from typing import Optional


class CaptionOverlay:
    def __init__(
        self,
        font_size: int = 26,
        max_lines: int = 3,
        position: str = "bottom",
        width_fraction: float = 0.8,
        opacity: float = 0.85,
        background_color: str = "#0B2545",
        text_color: str = "#FFFFFF",
    ):
        self.font_size = font_size
        self.max_lines = max_lines
        self.position = position
        self.width_fraction = width_fraction
        self.opacity = opacity
        self.background_color = background_color
        self.text_color = text_color

        self._root: Optional[tk.Tk] = None
        self._label: Optional[tk.Label] = None
        self._text_queue: "queue.Queue[str]" = queue.Queue()
        self._lines: list[str] = []

    def _build_window(self) -> None:
        root = tk.Tk()
        root.overrideredirect(True)       # borderless
        root.attributes("-topmost", True)  # always on top
        try:
            root.attributes("-alpha", self.opacity)
        except tk.TclError:
            pass  # alpha transparency not supported on this platform

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = int(screen_w * self.width_fraction)
        win_h = self.font_size * (self.max_lines + 1)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h - 60) if self.position == "bottom" else 40
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.configure(bg=self.background_color)

        label = tk.Label(
            root,
            text="",
            font=("Segoe UI", self.font_size),
            fg=self.text_color,
            bg=self.background_color,
            wraplength=win_w - 40,
            justify="center",
        )
        label.pack(expand=True, fill="both", padx=20, pady=10)

        self._root = root
        self._label = label
        self._poll_queue()

    def _poll_queue(self) -> None:
        try:
            while True:
                text = self._text_queue.get_nowait()
                self._apply_text(text)
        except queue.Empty:
            pass
        if self._root is not None:
            self._root.after(100, self._poll_queue)

    def _apply_text(self, text: str) -> None:
        self._lines.append(text)
        self._lines = self._lines[-self.max_lines :]
        if self._label is not None:
            self._label.config(text="\n".join(self._lines))

    def update(self, text: str) -> None:
        """Thread-safe: call this from any thread to push new caption text."""
        if text.strip():
            self._text_queue.put(text.strip())

    def run(self) -> None:
        """Blocking call — builds the window and starts the Tkinter mainloop.
        Call this from the main thread; run the audio/ASR pipeline on a
        background thread that calls `update()`."""
        self._build_window()
        assert self._root is not None
        self._root.mainloop()

    def close(self) -> None:
        if self._root is not None:
            self._root.after(0, self._root.destroy)
