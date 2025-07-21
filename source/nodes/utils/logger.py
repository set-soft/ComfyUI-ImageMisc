# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-ImageMisc
from __future__ import annotations  # Good practice
import logging
import os
import sys
from typing import Any, Callable
from .misc import NODES_NAME, NODES_DEBUG_VAR
from .comfy_notification import send_toast_notification


# 1. Initialize variables with the `Any` type.
#    This tells mypy not to make assumptions about their specific class.
Fore: Any
Back: Any
Style: Any

# 2. Perform the runtime import logic as before.
try:
    from colorama import init as colorama_init, Fore, Back, Style
    colorama_init()
except ImportError:
    # If colorama is not available, import our fallback.
    # mypy will now allow this assignment because the variables were declared as Any.
    from .ansi import Fore, Back, Style


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    def __init__(self):
        super(logging.Formatter, self).__init__()
        white = Fore.WHITE + Style.BRIGHT
        yellow = Fore.YELLOW + Style.BRIGHT
        red = Fore.RED + Style.BRIGHT
        red_alarm = Fore.RED + Back.WHITE + Style.BRIGHT
        cyan = Fore.CYAN + Style.BRIGHT
        reset = Style.RESET_ALL
        # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
        #          "(%(filename)s:%(lineno)d)"
        format = f"[{NODES_NAME} %(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
        format_simple = f"[{NODES_NAME}] %(message)s"

        self.FORMATS = {
            logging.DEBUG: cyan + format + reset,
            logging.INFO: white + format_simple + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: red_alarm + format + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def on_log_error_or_warning(record: logging.LogRecord) -> None:
    """
    This function is called whenever a log with level WARNING or higher is emitted.
    The 'record' object contains all information about the log event.
    """
    if record.levelno == logging.WARNING:
        summary = "Warning"
        severity = "warn"
    else:
        summary = "Error"
        severity = "error"
    send_toast_notification(record.getMessage(), summary=summary, severity=severity)


class WarningAndErrorFilter(logging.Filter):
    """
    A custom log filter that intercepts logs of a certain level.
    """
    def __init__(self, callback: Callable, level: int = logging.WARNING):
        """
        Initializes the filter.

        Args:
            callback: The function to call when a log record meets the level criteria.
            level: The minimum level to trigger the callback.
        """
        super().__init__()
        self._callback = callback
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """
        This method is called for every log record.
        """
        # Check if the log level is WARNING or higher
        if record.levelno >= self._level:
            self._callback(record)

        # Always return True to ensure the log is always processed
        # by the handlers after this filter.
        return True


# Create a new logger
logger = logging.getLogger(NODES_NAME)
logger.propagate = False

# Add the custom filter to the logger.
logger.addFilter(WarningAndErrorFilter(callback=on_log_error_or_warning))

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

# ######################
# Logger setup
# ######################
# 1. Determine the ComfyUI global log level (influenced by --verbose)
main_logger = logger
comfy_root_logger = logging.getLogger('comfy')
effective_comfy_level = logging.getLogger().getEffectiveLevel()
# 2. Check our custom environment variable for more verbosity
try:
    nodes_debug_env = int(os.environ.get(NODES_DEBUG_VAR, "0"))
except ValueError:
    nodes_debug_env = 0
# 3. Set node's logger level
if nodes_debug_env:
    main_logger.setLevel(logging.DEBUG - (nodes_debug_env - 1))
    final_level_str = f"DEBUG (due to {NODES_DEBUG_VAR}={nodes_debug_env})"
else:
    main_logger.setLevel(effective_comfy_level)
    final_level_str = logging.getLevelName(effective_comfy_level) + " (matching ComfyUI global)"
_initial_setup_logger = logging.getLogger(NODES_NAME + ".setup")  # A temporary logger for this message
_initial_setup_logger.debug(f"{NODES_NAME} logger level set to: {final_level_str}")
