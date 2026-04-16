"""
utils/face_auth.py
===================
InsightFace-based face registration and recognition.
Provides clean register() and identify() APIs used by run.py.
"""

import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError(
        "insightface is not installed.\n"
        "Run: pip install insightface onnxruntime"
    )


class FaceAuth:
    """
    Handles face registration (enrollment) and recognition.

    Usage:
        auth = FaceAuth()
        auth.capture_sample(frame)   # call 5 times during registration phase
        auth.finalize_registration() # locks the embedding

        result = auth.identify(frame)  # returns (bbox, sim) or None
    """

    # ── Config ────────────────────────────────────────────────────────────────
    MODEL_NAME      = "buffalo_s"     # fast MobileFaceNet - no GPU needed
    DET_SIZE        = (320, 320)      # detection resolution
    CTX_ID          = -1              # -1 = CPU
    MIN_CAPTURES    = 5               # samples needed for registration
    SIM_THRESHOLD   = 0.40            # cosine similarity cutoff
    DETECT_SCALE    = 0.5             # resize factor for face detection only

    def __init__(self):
        print("[FaceAuth] Loading InsightFace model...")
        self._app = FaceAnalysis(name=self.MODEL_NAME)
        self._app.prepare(ctx_id=self.CTX_ID, det_size=self.DET_SIZE)
        print("[FaceAuth] Ready.")

        self._samples: list = []          # collected embeddings during enroll
        self.registered_emb = None        # locked after finalize_registration()

    # ── Registration ──────────────────────────────────────────────────────────

    def capture_sample(self, bgr_frame: np.ndarray) -> bool:
        """
        Try to extract a face from bgr_frame and store its embedding as a sample.

        Returns True if a face was successfully captured, False otherwise.
        """
        faces = self._app.get(bgr_frame)
        if not faces:
            return False
        # Pick the largest detected face (closest to camera)
        biggest = max(faces, key=lambda f: _bbox_area(f.bbox))
        emb = biggest.normed_embedding
        self._samples.append(emb)
        return True

    def finalize_registration(self) -> bool:
        """
        Average all collected samples into one embedding.
        Returns True if at least MIN_CAPTURES samples exist.
        """
        if len(self._samples) < self.MIN_CAPTURES:
            return False
        self.registered_emb = np.mean(self._samples, axis=0)
        self.registered_emb /= np.linalg.norm(self.registered_emb)   # re-normalise
        self._samples.clear()
        return True

    def reset_registration(self):
        """Clear registration to allow re-enrollment."""
        self._samples.clear()
        self.registered_emb = None

    @property
    def is_registered(self) -> bool:
        return self.registered_emb is not None

    @property
    def samples_collected(self) -> int:
        return len(self._samples)

    # ── Inference ─────────────────────────────────────────────────────────────

    def identify(self, bgr_frame: np.ndarray):
        """
        Detect all faces and find the one that best matches the registered user.

        Parameters
        ----------
        bgr_frame : np.ndarray  (full-resolution BGR frame)

        Returns
        -------
        (bbox [x1,y1,x2,y2], similarity float)   if target found
        None                                       if no match
        """
        if not self.is_registered:
            return None

        # Downscale for faster detection
        small = _resize(bgr_frame, self.DETECT_SCALE)
        faces = self._app.get(small)

        if not faces:
            return None

        best_bbox, best_sim = None, -1.0
        scale_inv = 1.0 / self.DETECT_SCALE

        for face in faces:
            sim = float(np.dot(face.normed_embedding, self.registered_emb))
            if sim > best_sim:
                best_sim = sim
                # Scale bbox back to original resolution
                best_bbox = (face.bbox * scale_inv).astype(int)

        if best_sim >= self.SIM_THRESHOLD:
            return best_bbox, best_sim
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_area(bbox) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)))
