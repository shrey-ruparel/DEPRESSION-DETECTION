"""
quiz.py — thin wrapper so Flask app can do:
    from modules.quiz import predict_result
"""
import sys
import os

# Add BACKEND/MODULES to path so the real quiz.py can load its model
_backend_modules = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "BACKEND", "MODULES"
)
if _backend_modules not in sys.path:
    sys.path.insert(0, _backend_modules)

from quiz import predict_result, questions  # noqa: F401
