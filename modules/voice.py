"""
voice.py — thin wrapper so Flask app can do:
    from modules.voice import analyze_audio_file
"""
import sys
import os

# Add BACKEND/MODULES to path
_backend_modules = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "BACKEND", "MODULES"
)
if _backend_modules not in sys.path:
    sys.path.insert(0, _backend_modules)

from voice_nlp import assess_depression  # noqa: F401
