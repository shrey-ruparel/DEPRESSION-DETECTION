"""
Depression Detection — Multimodal Fusion  (v2)
===============================================
Runs face analysis (DeepFace) and NLP voice analysis TOGETHER:
  • Camera window stays open throughout the entire questionnaire
  • Questions are asked one-by-one; each answer is transcribed via Whisper
  • Face snapshots are collected the whole time (not just during answers)
  • After all questions, both scores are fused into a final depression level

Files required in the same folder:
  face.py        — FaceAnalyzer, FaceAnalysisResult, extract_fusion_payload
  voice_nlp.py   — assess_depression
  voice_stt.py   — run_questionnaire   (your existing STT / Whisper module)

Run:
  python fusion.py
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import threading
import time
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Local modules — these must be in the same directory
from face import FaceAnalyzer, FaceAnalysisResult, extract_fusion_payload
from voice_nlp import assess_depression
from voice_stt import run_questionnaire        # your Whisper STT module


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

FACE_WEIGHT  = 0.35    # face (involuntary micro-expressions) — less gameable
VOICE_WEIGHT = 0.65    # voice/NLP (semantic meaning) — stronger clinical signal

CAMERA_WARMUP_TIMEOUT = 15   # seconds to wait for DeepFace warmup before giving up
FONT  = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY  = (180, 180, 180)
GREEN = (50, 220, 80)
RED   = (0, 60, 220)
AMBER = (0, 160, 255)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def face_score_to_depression(avg_face_score: float) -> float:
    """Face module gives a POSITIVITY score (0=bad, 1=good). Flip it."""
    return round(1.0 - avg_face_score, 4)


def get_depression_level(fused_score: float) -> tuple:
    """Return (level_str, action_str) from a 0–1 fused depression score."""
    if fused_score < 0.30:
        return ("Normal",   "No immediate concern. Continue monitoring.")
    elif fused_score < 0.55:
        return ("Moderate", "Consider speaking to a counselor or trusted person.")
    else:
        return ("Severe",   "Professional consultation strongly recommended. Please reach out for help.")


def wrap_text(text: str, max_chars: int = 55) -> list:
    """Break a long string into lines of max_chars for cv2 rendering."""
    words  = text.split()
    lines  = []
    line   = ""
    for w in words:
        if len(line) + len(w) + 1 <= max_chars:
            line = (line + " " + w).strip()
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


# ─────────────────────────────────────────────────────────────
# Result Dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    # Face
    avg_face_score:             float = 0.0
    face_depression_score:      float = 0.0
    face_dominant_emotion:      str   = "neutral"
    face_emotion_distribution:  dict  = field(default_factory=dict)
    face_detected_ratio:        float = 0.0
    face_snapshots:             int   = 0

    # Voice / NLP
    voice_normalized_score:     float = 0.0
    voice_depression_level:     str   = "Normal"
    voice_total_weighted:       float = 0.0
    voice_per_question:         list  = field(default_factory=list)

    # Fusion
    fused_score:                float = 0.0
    final_depression_level:     str   = "Normal"
    recommended_action:         str   = ""
    face_weight_used:           float = FACE_WEIGHT
    voice_weight_used:          float = VOICE_WEIGHT


# ─────────────────────────────────────────────────────────────
# Shared State (written by face thread, read by display thread)
# ─────────────────────────────────────────────────────────────

class SharedState:
    def __init__(self):
        self._lock           = threading.Lock()
        self.emotion         = "Warming up..."
        self.face_score      = 0.0
        self.avg_score       = 0.0
        self.snapshot_count  = 0
        self.color           = GRAY
        self.current_question = ""
        self.status_msg      = "Starting..."
        self.camera_ready    = False   # set True once DeepFace warms up
        self.all_done        = False   # set True to close window

    # Thread-safe getters / setters
    def update_face(self, emotion, face_score, avg_score, count, color):
        with self._lock:
            self.emotion        = emotion
            self.face_score     = face_score
            self.avg_score      = avg_score
            self.snapshot_count = count
            self.color          = color

    def set_question(self, text):
        with self._lock:
            self.current_question = text

    def set_status(self, text):
        with self._lock:
            self.status_msg = text

    def get_snapshot(self):
        with self._lock:
            return (
                self.emotion, self.face_score, self.avg_score,
                self.snapshot_count, self.color,
                self.current_question, self.status_msg
            )


# ─────────────────────────────────────────────────────────────
# Main Detector
# ─────────────────────────────────────────────────────────────

class DepressionDetector:
    """
    Orchestrates the full session:
      1. Opens the camera window (main thread — required by OpenCV)
      2. Starts face analysis loop (background thread)
      3. Starts questionnaire (separate background thread using voice_stt)
      4. After questionnaire finishes, stops face analysis
      5. Fuses results and prints the final report
    """

    def __init__(self, session_id: str = "session_001"):
        self.session_id       = session_id
        self.state            = SharedState()

        self._face_result:    Optional[FaceAnalysisResult] = None
        self._voice_responses: Optional[list]              = None
        self._voice_result:   Optional[dict]               = None
        self._face_error:     Optional[str]                = None
        self._voice_error:    Optional[str]                = None

        self._face_analyzer:  Optional[FaceAnalyzer]       = None
        self._voice_done:     threading.Event              = threading.Event()
        self._face_ready:     threading.Event              = threading.Event()

    # ── Entry Point ────────────────────────────────────────────

    def run(self) -> FusionResult:
        print("\n" + "=" * 65)
        print("  DEPRESSION DETECTION — Multimodal Fusion v2")
        print("  Camera (DeepFace) + Voice (Whisper + NLP)")
        print("=" * 65)
        print("\n[INFO] Camera window will open. Questionnaire starts after warmup.\n")

        # Build face analyzer (headless=False: WE control the cv2 window here)
        self._face_analyzer = FaceAnalyzer(
            session_id=self.session_id,
            snapshot_interval=2.0,
            detector_backend="opencv",
            save_face_crops=False,
            frame_skip=2,
        )

        # --- Open camera and grab a test frame BEFORE starting threads ---
        self._face_analyzer._open_camera()
        cap = self._face_analyzer._cap

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            raise RuntimeError(
                "Camera opened but cannot read frames.\n"
                "Close Teams / Zoom / OBS and retry."
            )
        h, w = test_frame.shape[:2]
        print(f"[Camera] {w}×{h} — OK")

        # --- Thread 1: DeepFace analysis loop (background) ---
        face_thread = threading.Thread(
            target=self._face_loop,
            args=(cap,),
            daemon=True,
        )
        face_thread.start()

        # --- Thread 2: Questionnaire (background — needs terminal I/O) ---
        voice_thread = threading.Thread(
            target=self._voice_loop,
            daemon=True,
        )

        # --- Main thread: cv2 display loop ---
        self._display_loop(cap, face_thread, voice_thread)

        # --- After window closes: NLP analysis ---
        if self._voice_responses:
            print("\n[NLP] Analysing voice responses...")
            self.state.set_status("Running NLP analysis...")
            try:
                self._voice_result = assess_depression(self._voice_responses)
            except Exception as e:
                self._voice_error = str(e)
                print(f"[NLP Error] {e}")
        else:
            print("[Voice] No responses captured.")

        # --- Fuse and return ---
        return self._fuse()

    # ── Display Loop (main thread) ──────────────────────────────

    def _display_loop(
        self,
        cap: cv2.VideoCapture,
        face_thread: threading.Thread,
        voice_thread: threading.Thread,
    ) -> None:
        """
        Runs on the main thread.
        Shows live camera feed with overlays.
        Starts voice_thread once camera is warmed up.
        Exits when questionnaire finishes (or user presses Q).
        """
        voice_started = False
        window_name   = "Depression Detector  —  Press Q to abort"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # Start questionnaire once face is ready
            if self.state.camera_ready and not voice_started:
                print("[Fusion] Camera ready. Starting questionnaire...\n")
                self.state.set_status("Questionnaire started — answer each question aloud.")
                voice_thread.start()
                voice_started = True

            # Draw overlay on frame
            self._draw_overlay(frame)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[Fusion] Q pressed — aborting session.")
                self._voice_done.set()
                break

            # Exit when questionnaire is done
            if voice_started and self._voice_done.is_set():
                time.sleep(0.5)   # show final frame briefly
                break

        # --- Cleanup ---
        self._face_analyzer._running = False
        face_thread.join(timeout=6)
        self._face_result = self._face_analyzer.stop()

        cap.release()
        cv2.destroyAllWindows()

    # ── Face Analysis Loop (background thread) ──────────────────

    def _face_loop(self, cap: cv2.VideoCapture) -> None:
        """
        Runs on a background thread.
        Calls DeepFace every snapshot_interval seconds.
        Updates SharedState so the display thread can overlay the results.
        """
        from deepface import DeepFace
        from face import EMOTION_SCORES
        from collections import deque

        analyzer     = self._face_analyzer
        interval     = analyzer.snapshot_interval
        emotion_buf  = deque(maxlen=5)
        last_snap    = time.time()
        start_time   = time.time()

        DEPRESSIVE_EMOTIONS = {"sad", "fear", "disgust", "angry"}

        # Warmup
        print("[Face] Loading DeepFace models (first run may take ~10s)...")
        self.state.set_status("Loading face models — please wait...")
        try:
            ret, wf = cap.read()
            if ret and wf is not None:
                DeepFace.analyze(
                    img_path=wf,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True,
                )
            print("[Face] DeepFace models loaded OK.")
        except Exception as e:
            print(f"[Face] Warmup note: {e}")

        self.state.camera_ready = True
        self._face_ready.set()
        analyzer._running    = True
        analyzer._start_time = time.time()

        frame_idx   = 0
        total_frms  = 0
        face_frms   = 0

        while analyzer._running:
            now = time.time()
            if (now - last_snap) < interval:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            total_frms += 1
            frame_idx  += 1

            try:
                results = DeepFace.analyze(
                    img_path=frame.copy(),
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=True,
                    silent=True,
                )
                analysis     = results[0] if isinstance(results, list) else results
                raw_emotions = analysis["emotion"]
                dominant     = analysis["dominant_emotion"]
                region       = analysis.get("region", {})

                total_e      = sum(raw_emotions.values()) or 1.0
                norm_e       = {k: v / total_e for k, v in raw_emotions.items()}

                emotion_buf.append(norm_e)
                smoothed = {
                    k: sum(e.get(k, 0.0) for e in emotion_buf) / len(emotion_buf)
                    for k in norm_e
                }

                face_score = round(
                    min(max(
                        sum(EMOTION_SCORES.get(e, 0.5) * p for e, p in smoothed.items()),
                        0.0), 1.0),
                    4
                )
                confidence = round(smoothed.get(dominant, 0.0), 4)

                from face import EmotionSnapshot
                snapshot = EmotionSnapshot(
                    timestamp     = round(now - analyzer._start_time, 3),
                    dominant_emotion = dominant,
                    emotion_scores   = {k: round(v, 2) for k, v in raw_emotions.items()},
                    normalized_emotions = {k: round(v, 4) for k, v in smoothed.items()},
                    face_score    = face_score,
                    confidence    = confidence,
                    frame_index   = frame_idx,
                )

                with analyzer._lock:
                    analyzer._snapshots.append(snapshot)
                    avg = sum(s.face_score for s in analyzer._snapshots) / len(analyzer._snapshots)
                    analyzer._overlay_avg     = round(avg, 4)
                    analyzer._overlay_emotion = dominant
                    analyzer._overlay_score   = face_score
                    analyzer._overlay_count   = len(analyzer._snapshots)
                    analyzer._overlay_color   = FaceAnalyzer._emotion_color(dominant)
                    analyzer._face_detected_frames += 1
                    analyzer._total_frames   = total_frms
                    analyzer._frame_index    = frame_idx

                # Update shared state for overlay rendering
                color_map = {
                    "happy":    GREEN,
                    "surprise": (0, 200, 255),
                    "neutral":  GRAY,
                    "fear":     AMBER,
                    "sad":      (200, 100, 50),
                    "angry":    RED,
                    "disgust":  (100, 0, 200),
                }
                self.state.update_face(
                    emotion    = dominant,
                    face_score = face_score,
                    avg_score  = round(avg, 4),
                    count      = len(analyzer._snapshots),
                    color      = color_map.get(dominant, GRAY),
                )
                face_frms  += 1
                last_snap   = now

                print(
                    f"  [Face #{len(analyzer._snapshots):>3}] "
                    f"{dominant.upper():<10} score={face_score:.3f}  avg={avg:.3f}"
                )

            except Exception:
                # No face detected this frame — just skip
                last_snap = now

        print("[Face] Analysis loop exited.")

    # ── Voice / Questionnaire Loop (background thread) ──────────

    def _voice_loop(self) -> None:
        """
        Runs on a background thread.
        Calls run_questionnaire() from voice_stt.py — blocking.
        Sets _voice_done when finished so the display thread can exit.
        """
        try:
            print("[Voice] Questionnaire starting...\n")
            self._voice_responses = run_questionnaire(
                question_callback=self.state.set_question,   # pass each Q to overlay
                status_callback=self.state.set_status,
            )
            print("\n[Voice] Questionnaire complete.")
            self.state.set_question("")
            self.state.set_status("Voice done. Stopping camera...")
        except TypeError:
            # voice_stt.run_questionnaire() doesn't accept callbacks — run plain
            print("[Voice] Note: voice_stt does not support callbacks — running without overlay updates.")
            self._voice_responses = run_questionnaire()
            print("\n[Voice] Questionnaire complete.")
            self.state.set_status("Voice done. Stopping camera...")
        except Exception as e:
            self._voice_error = str(e)
            print(f"[Voice Error] {e}")
            traceback.print_exc()
        finally:
            self._voice_done.set()   # always signal done, even on error

    # ── Overlay Renderer ────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray) -> None:
        """
        Draws a two-part overlay:
          • Bottom bar  — live face analysis stats
          • Top banner  — current question (if any)
        """
        h, w = frame.shape[:2]
        (
            emotion, face_score, avg_score,
            snap_count, color,
            question, status
        ) = self.state.get_snapshot()

        # ── Bottom bar (face stats) ──────────────────────────────
        bar_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame,
            f"Emotion: {emotion.upper()}",
            (12, h - 62), FONT, 0.68, color, 2)

        cv2.putText(frame,
            f"Frame Score: {face_score:.3f}  |  Avg Score: {avg_score:.3f}",
            (12, h - 38), FONT, 0.52, GRAY, 1)

        cv2.putText(frame,
            f"Snapshots: {snap_count}   |   {status}",
            (12, h - 14), FONT, 0.42, (130, 130, 130), 1)

        # ── Score bar ────────────────────────────────────────────
        bar_full = w - 24
        bar_fill = int(avg_score * bar_full)
        bar_y    = h - bar_h - 8
        cv2.rectangle(frame, (12, bar_y - 6), (12 + bar_full, bar_y), (40, 40, 40), -1)
        bar_color = GREEN if avg_score > 0.55 else (AMBER if avg_score > 0.35 else RED)
        cv2.rectangle(frame, (12, bar_y - 6), (12 + bar_fill, bar_y), bar_color, -1)

        # ── Top banner (current question) ────────────────────────
        if question:
            lines      = wrap_text(question, max_chars=60)
            banner_h   = 30 + 26 * len(lines)
            top_overlay = frame.copy()
            cv2.rectangle(top_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(top_overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, "QUESTION", (12, 22), FONT, 0.5, AMBER, 1)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 44 + i * 26), FONT, 0.6, WHITE, 1)

    # ── Fusion ──────────────────────────────────────────────────

    def _fuse(self) -> FusionResult:
        result = FusionResult()

        # ── Face values ───────────────────────────────────────────
        fr = self._face_result
        if fr and fr.snapshots:
            result.avg_face_score            = fr.average_face_score
            result.face_depression_score     = face_score_to_depression(fr.average_face_score)
            result.face_dominant_emotion     = fr.dominant_emotion_overall
            result.face_emotion_distribution = fr.emotion_distribution
            result.face_detected_ratio       = fr.face_detected_ratio
            result.face_snapshots            = len(fr.snapshots)
        else:
            print("[Fusion] No valid face data — using neutral fallback (0.5).")
            result.face_depression_score = 0.5

        # ── Voice values ──────────────────────────────────────────
        vr = self._voice_result
        if vr:
            result.voice_normalized_score = vr["normalized_score"]
            result.voice_depression_level = vr["depression_level"]
            result.voice_total_weighted   = vr["total_weighted_score"]
            result.voice_per_question     = vr["per_question"]
        else:
            print("[Fusion] No valid voice data — using neutral fallback (0.5).")
            result.voice_normalized_score = 0.5

        # ── Determine weights ─────────────────────────────────────
        face_w  = FACE_WEIGHT
        voice_w = VOICE_WEIGHT

        has_face  = bool(fr and fr.snapshots)
        has_voice = bool(vr)

        if has_face and not has_voice:
            face_w, voice_w = 1.0, 0.0
        elif has_voice and not has_face:
            face_w, voice_w = 0.0, 1.0
        # else use defaults

        # ── Weighted fusion ───────────────────────────────────────
        fused = (result.face_depression_score * face_w) + (result.voice_normalized_score * voice_w)
        fused = round(min(max(fused, 0.0), 1.0), 4)

        result.fused_score       = fused
        result.face_weight_used  = face_w
        result.voice_weight_used = voice_w

        level, action = get_depression_level(fused)
        result.final_depression_level = level
        result.recommended_action     = action

        return result


# ─────────────────────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────────────────────

def print_report(result: FusionResult) -> None:
    W   = 65
    SEP = "=" * W
    DIV = "  " + "─" * (W - 2)

    print(f"\n{SEP}")
    print("  FINAL FUSION REPORT")
    print(SEP)

    # ── Face ─────────────────────────────────────────────────────
    print(f"\n  FACE ANALYSIS")
    print(DIV)
    print(f"  Positivity Score      : {result.avg_face_score:.4f}")
    print(f"  Depression Indicator  : {result.face_depression_score:.4f}  (1 − face score)")
    print(f"  Dominant Emotion      : {result.face_dominant_emotion.upper()}")
    print(f"  Snapshots Captured    : {result.face_snapshots}")
    print(f"  Face Detected Ratio   : {result.face_detected_ratio:.1%}")
    if result.face_emotion_distribution:
        print("  Emotion Distribution  :")
        for emo, prob in sorted(result.face_emotion_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 25)
            print(f"    {emo:<12} {bar:<25} {prob:.3f}")

    # ── Voice ────────────────────────────────────────────────────
    print(f"\n  VOICE / NLP ANALYSIS")
    print(DIV)
    print(f"  Depression Score      : {result.voice_normalized_score:.4f}")
    print(f"  Voice Level           : {result.voice_depression_level}")
    print(f"  Total Weighted Score  : {result.voice_total_weighted:.3f}")

    # ── Fusion ───────────────────────────────────────────────────
    print(f"\n  FUSION")
    print(DIV)
    print(f"  Face Weight           : {result.face_weight_used:.0%}")
    print(f"  Voice Weight          : {result.voice_weight_used:.0%}")
    print(f"  Fused Score           : {result.fused_score:.4f}  ({result.fused_score * 100:.1f}%)")

    bar_len  = int(result.fused_score * 40)
    bar      = "█" * bar_len + "░" * (40 - bar_len)
    icon_map = {"Normal": "✅", "Moderate": "⚠️ ", "Severe": "🚨"}
    icon     = icon_map.get(result.final_depression_level, "")

    print(f"\n  [{bar}] {result.fused_score * 100:.1f}%")
    print(f"\n  {icon} Depression Level     : {result.final_depression_level.upper()}")
    print(f"  Recommended Action    : {result.recommended_action}")
    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────
# JSON Export
# ─────────────────────────────────────────────────────────────

def export_json(result: FusionResult, path: str = "fusion_result.json") -> None:
    data = {
        "face": {
            "avg_face_score":        result.avg_face_score,
            "face_depression_score": result.face_depression_score,
            "dominant_emotion":      result.face_dominant_emotion,
            "emotion_distribution":  result.face_emotion_distribution,
            "face_detected_ratio":   result.face_detected_ratio,
            "snapshots_captured":    result.face_snapshots,
        },
        "voice": {
            "normalized_score":     result.voice_normalized_score,
            "depression_level":     result.voice_depression_level,
            "total_weighted_score": result.voice_total_weighted,
        },
        "fusion": {
            "fused_score":            result.fused_score,
            "face_weight":            result.face_weight_used,
            "voice_weight":           result.voice_weight_used,
            "final_depression_level": result.final_depression_level,
            "recommended_action":     result.recommended_action,
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Fusion] Result saved → {path}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = DepressionDetector(session_id="session_001")
    try:
        result = detector.run()
    except KeyboardInterrupt:
        print("\n[Fusion] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Fusion] Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print_report(result)
    export_json(result)