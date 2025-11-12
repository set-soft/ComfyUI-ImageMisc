# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-ImageMisc
from seconohe.logger import initialize_logger

__version__ = "1.4.0"
NODES_NAME = "ImageMisc"
main_logger = initialize_logger(NODES_NAME)

# Epsilon: small value to avoid "divide by 0" errors
EPS = 1e-8
F_POINTS = 255
