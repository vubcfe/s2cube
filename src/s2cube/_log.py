# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Progress messages, flushed immediately so they appear in redirected logs."""
import functools

log_print = functools.partial(print, flush=True)
