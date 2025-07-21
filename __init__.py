# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-ImageMisc
from .source.nodes import nodes_img
import inspect
import logging
from .source.nodes.utils.misc import NODES_NAME

init_logger = logging.getLogger(NODES_NAME + ".__init__")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for name, obj in inspect.getmembers(nodes_img):
    # We skip nodes imported from the ComfyUI main nodes
    if not inspect.isclass(obj) or not hasattr(obj, "INPUT_TYPES") or obj.__module__ == "nodes":
        continue
    assert hasattr(obj, "UNIQUE_NAME"), f"No name for {obj.__name__}"
    NODE_CLASS_MAPPINGS[obj.UNIQUE_NAME] = obj
    NODE_DISPLAY_NAME_MAPPINGS[obj.UNIQUE_NAME] = obj.DISPLAY_NAME

init_logger.info(f"Registering {len(NODE_CLASS_MAPPINGS)} node(s).")
init_logger.debug(f"{list(NODE_DISPLAY_NAME_MAPPINGS.values())}")

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
