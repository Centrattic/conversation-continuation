import sys
from pathlib import Path
from typing import Any


class ConversationLogger:
    """A simple logger that writes to a file and to the console."""

    COLORS = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
    }

    def __init__(self, log_path: Path | None = None):
        self.terminal = sys.stdout
        # Open the file and keep the handle
        if log_path:
            self.log_file = open(log_path, "w", encoding='utf-8')
        else:
            self.log_file = None

    def log_to_all(self, message: Any, color: str = ""):
        """Writes a message to the console and the log file."""

        if color in self.COLORS:
            code = self.COLORS[color]
            reset = self.COLORS["reset"]
            message = f"{code}{message}{reset}"

        print(message, file=self.terminal, flush=True)
        if self.log_file:
            print(message, file=self.log_file, flush=True)

    def log_to_file(self, message: Any):
        """Writes a message to the console and the log file."""
        if self.log_file:
            print(message, file=self.log_file, flush=True)
        # else:
        #     print("No log file exists. Nothing was saved.",file=self.terminal, flush=True)

    def log_to_console(self, message: Any):
        """Writes a message to the console and the log file."""
        print(message, file=self.terminal, flush=True)

    def close(self):
        """Closes the log file handle."""
        if self.log_file:
            self.log_file.close()


# Add colors
