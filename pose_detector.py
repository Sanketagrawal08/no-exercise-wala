from dataclasses import dataclass
import math
from typing import Optional, Dict, Tuple
import os
import sys
import contextlib
import logging

# Reduce TensorFlow/absl/MediaPipe noisy logs; set BEFORE importing heavy libs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = warning+, '3' = error only
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@contextlib.contextmanager
def _suppress_stderr():
    """Context manager that temporarily redirects stderr to devnull to suppress noisy C++ logs."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Landmarks we'll check (face / nose removed because not required for leg exercises)
REQUIRED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]


@dataclass
class PostureResult:
    is_full_body: bool
    is_upright: bool
    shoulder_hip_error: float
    spine_angle_deg: float


class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        # Suppress noisy stderr output from underlying C++ libraries during initialization
        with _suppress_stderr():
            self.pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence,
                                     model_complexity=1)

    def process(self, frame: np.ndarray) -> Tuple[Optional[mp.solutions.pose.PoseLandmark], np.ndarray]:
        """Process a BGR frame (OpenCV) and return landmarks + annotated frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        annotated = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )
        return results.pose_landmarks, annotated

    def landmarks_to_array(self, landmarks, image_shape) -> Dict[str, Tuple[float,float,float]]:
        """Convert mediapipe landmarks to a dict of (x,y,visibility) in pixel coords."""
        h, w = image_shape[:2]
        lm = {}
        for i, l in enumerate(landmarks.landmark):
            lm[mp_pose.PoseLandmark(i).name] = (l.x * w, l.y * h, l.visibility)
        return lm

    def is_full_body(self, landmarks, image_shape, visibility_threshold: float = 0.5) -> bool:
        if landmarks is None:
            return False
        lm = self.landmarks_to_array(landmarks, image_shape)
        # Check required landmarks exist and are visible
        for req in REQUIRED_LANDMARKS:
            name = req.name
            if name not in lm or lm[name][2] < visibility_threshold:
                return False
        # Simple standing check: ankle lower than knee lower than hip (y increases downward)
        left_knee_y = lm['LEFT_KNEE'][1]
        right_knee_y = lm['RIGHT_KNEE'][1]
        left_ankle_y = lm['LEFT_ANKLE'][1]
        right_ankle_y = lm['RIGHT_ANKLE'][1]
        left_hip_y = lm['LEFT_HIP'][1]
        right_hip_y = lm['RIGHT_HIP'][1]
        knees_below_hips = (left_knee_y > left_hip_y) and (right_knee_y > right_hip_y)
        ankles_below_knees = (left_ankle_y > left_knee_y) and (right_ankle_y > right_knee_y)
        return knees_below_hips and ankles_below_knees

    def posture_metrics(self, landmarks, image_shape) -> Optional[PostureResult]:
        """Compute posture metrics (shoulder-hip alignment and spine angle).

        Returns None if landmarks missing.
        """
        if landmarks is None:
            return None
        lm = self.landmarks_to_array(landmarks, image_shape)
        # Midpoints
        left_sh = np.array(lm['LEFT_SHOULDER'][:2])
        right_sh = np.array(lm['RIGHT_SHOULDER'][:2])
        left_hp = np.array(lm['LEFT_HIP'][:2])
        right_hp = np.array(lm['RIGHT_HIP'][:2])
        shoulder_mid = (left_sh + right_sh) / 2.0
        hip_mid = (left_hp + right_hp) / 2.0

        # Torso height estimate
        torso_height = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_height < 1e-6:
            return None

        # Horizontal offset between shoulder_mid and hip_mid
        horizontal_offset = abs(shoulder_mid[0] - hip_mid[0])
        shoulder_hip_error = horizontal_offset / torso_height  # normalized

        # Spine angle w.r.t vertical: angle of vector hip->shoulder
        vec = shoulder_mid - hip_mid  # y increases downward
        # angle from vertical (in radians): arctan(dx/dy)
        spine_angle_rad = math.atan2(vec[0], vec[1]) if vec[1] != 0 else math.copysign(math.pi/2, vec[0])
        spine_angle_deg = abs(math.degrees(spine_angle_rad))

        # Determine upright with thresholds (tunable)
        # shoulder_hip_error < 0.12 (12% of torso height) and spine_angle < 12 deg and shoulders roughly level
        shoulders_level = abs(left_sh[1] - right_sh[1]) / torso_height < 0.12
        is_upright = (shoulder_hip_error < 0.12) and (spine_angle_deg < 12) and shoulders_level
        return PostureResult(is_full_body=True, is_upright=is_upright,
                             shoulder_hip_error=shoulder_hip_error, spine_angle_deg=spine_angle_deg)

    def detect_bird_dog(self, landmarks, image_shape,
                         horiz_threshold_deg: float = 25.0,
                         angle_tolerance_deg: float = 15.0,
                         opposite_dot_threshold: float = -0.2,
                         min_extension_ratio: float = 0.28,
                         torso_angle_thresh: float = 35.0,
                         visibility_threshold: float = 0.35) -> Optional[Dict]:
        """Detect classic kneeling Bird Dog extension with partial-visibility support.

        This version allows using elbow/knee as fallbacks when wrist/ankle are not visible.
        A limb is accepted if either its distal joint (wrist/ankle) or intermediate joint
        (elbow/knee) is visible and shows sufficient extension. Small crookedness (5–15°)
        is tolerated via angle_tolerance_deg and average-angle allowance.
        """
        if landmarks is None:
            return None
        lm = self.landmarks_to_array(landmarks, image_shape)
        # require torso anchors
        for r in ['LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_HIP','RIGHT_HIP']:
            if r not in lm:
                return None

        # helper to choose endpoint: prefer distal (wrist/ankle) if visible else intermediate (elbow/knee)
        def choose_point(distal_name, intermediate_name):
            d = lm.get(distal_name)
            i = lm.get(intermediate_name)
            if d is not None and d[2] >= visibility_threshold:
                return d[:2]
            if i is not None and i[2] >= visibility_threshold:
                return i[:2]
            return None

        left_sh = np.array(lm['LEFT_SHOULDER'][:2])
        right_sh = np.array(lm['RIGHT_SHOULDER'][:2])
        left_hp = np.array(lm['LEFT_HIP'][:2])
        right_hp = np.array(lm['RIGHT_HIP'][:2])
        shoulder_mid = (left_sh + right_sh) / 2.0
        hip_mid = (left_hp + right_hp) / 2.0
        torso_vec = shoulder_mid - hip_mid  # y increases downward
        torso_height = np.linalg.norm(torso_vec)
        if torso_height < 1e-6:
            return None
        torso_angle_rad = math.atan2(torso_vec[0], torso_vec[1]) if torso_vec[1] != 0 else math.copysign(math.pi/2, torso_vec[0])
        torso_angle_deg = abs(math.degrees(torso_angle_rad))

        # prepare endpoints with visibility fallbacks
        left_arm_end = choose_point('LEFT_WRIST', 'LEFT_ELBOW')
        right_arm_end = choose_point('RIGHT_WRIST', 'RIGHT_ELBOW')
        left_leg_end = choose_point('LEFT_ANKLE', 'LEFT_KNEE')
        right_leg_end = choose_point('RIGHT_ANKLE', 'RIGHT_KNEE')

        def vec(a, b):
            return np.array(b) - np.array(a)

        def angle_from_horizontal(v):
            if np.linalg.norm(v) < 1e-6:
                return 90.0
            ang = abs(math.degrees(math.atan2(v[1], v[0])))
            return min(ang, 180 - ang)

        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        combos = []
        if left_arm_end is not None and right_leg_end is not None:
            combos.append({'side': 'left', 'arm_base': left_sh, 'arm_end': left_arm_end, 'leg_base': right_hp, 'leg_end': right_leg_end})
        if right_arm_end is not None and left_leg_end is not None:
            combos.append({'side': 'right', 'arm_base': right_sh, 'arm_end': right_arm_end, 'leg_base': left_hp, 'leg_end': left_leg_end})

        best = None
        for c in combos:
            arm_vec = vec(c['arm_base'], c['arm_end'])
            leg_vec = vec(c['leg_base'], c['leg_end'])
            arm_angle = angle_from_horizontal(arm_vec)
            leg_angle = angle_from_horizontal(leg_vec)
            arm_len = np.linalg.norm(arm_vec)
            leg_len = np.linalg.norm(leg_vec)
            arm_ratio = arm_len / torso_height
            leg_ratio = leg_len / torso_height

            arm_base_ok = (arm_angle <= horiz_threshold_deg) and (arm_ratio >= min_extension_ratio)
            leg_base_ok = (leg_angle <= horiz_threshold_deg) and (leg_ratio >= min_extension_ratio)
            arm_relaxed = (arm_angle <= horiz_threshold_deg + angle_tolerance_deg) and (arm_ratio >= (min_extension_ratio * 0.85))
            leg_relaxed = (leg_angle <= horiz_threshold_deg + angle_tolerance_deg) and (leg_ratio >= (min_extension_ratio * 0.85))

            avg_angle = (arm_angle + leg_angle) / 2.0
            avg_ok = avg_angle <= (horiz_threshold_deg + angle_tolerance_deg)

            uarm = unit(arm_vec)
            uleg = unit(leg_vec)
            dot = np.dot(uarm, uleg)
            opposite_ok = dot <= opposite_dot_threshold

            allowed_extension = ((arm_base_ok and leg_base_ok) or (arm_base_ok and leg_relaxed) or (leg_base_ok and arm_relaxed) or avg_ok)

            is_ext = allowed_extension and opposite_ok and (torso_angle_deg <= torso_angle_thresh)

            c.update({'arm_angle': arm_angle, 'leg_angle': leg_angle, 'arm_ratio': arm_ratio, 'leg_ratio': leg_ratio, 'dot': dot, 'is_extended': is_ext, 'torso_angle': torso_angle_deg})
            if best is None or (c['is_extended'] and not best.get('is_extended', False)):
                best = c

        return best

    def detect_reverse_leg_raise(self, landmarks, image_shape,
                             min_lift_ratio: float = 0.06,      # 👈 loose (lowered to be more permissive)
                             hip_tilt_ratio_max: float = 0.12, # 👈 loose
                             visibility_threshold: float = 0.25):
        """
        Loose Reverse Leg Raise detection:
        - Only hips + legs
        - Small lift is enough
        - Hip must remain mostly stable
        """
        if landmarks is None:
            return None

        lm = self.landmarks_to_array(landmarks, image_shape)

        # Required: hips
        for k in ['LEFT_HIP', 'RIGHT_HIP']:
            if k not in lm or lm[k][2] < visibility_threshold:
                return None

        left_hip = np.array(lm['LEFT_HIP'][:2])
        right_hip = np.array(lm['RIGHT_HIP'][:2])

        torso_height = np.linalg.norm(left_hip - right_hip)
        if torso_height < 1e-6:
            return None

        # helper: ankle > knee fallback
        def leg_point(ankle, knee):
            if ankle in lm and lm[ankle][2] >= visibility_threshold:
                return np.array(lm[ankle][:2])
            if knee in lm and lm[knee][2] >= visibility_threshold:
                return np.array(lm[knee][:2])
            return None

        left_leg = leg_point('LEFT_ANKLE', 'LEFT_KNEE')
        right_leg = leg_point('RIGHT_ANKLE', 'RIGHT_KNEE')

        if left_leg is None or right_leg is None:
            return None

        # how much leg is lifted
        left_lift = left_hip[1] - left_leg[1]
        right_lift = right_hip[1] - right_leg[1]

        left_ratio = left_lift / torso_height
        right_ratio = right_lift / torso_height

        left_up = left_ratio > min_lift_ratio
        right_up = right_ratio > min_lift_ratio

        # exactly one leg should be lifted
        if left_up == right_up:
            return None

        # hip stability check (VERY IMPORTANT)
        hip_diff = abs(left_hip[1] - right_hip[1])
        hip_tilt_ratio = hip_diff / torso_height

        if hip_tilt_ratio > hip_tilt_ratio_max:
            return None

        side = "left" if left_up else "right"

        return {
            "side": side,
            "is_lifted": True,
            "lift_ratio": left_ratio if left_up else right_ratio,
            "hip_tilt_ratio": hip_tilt_ratio
        }

    def detect_shoulder_raise(self, landmarks, image_shape,
                              up_threshold: float = 0.15,
                              down_threshold: float = 0.05,
                              visibility_threshold: float = 0.4):
        """
        Detect shoulder raise with joined hands (down → up → down).

        Logic:
        - Only shoulders + wrists
        - Hands move from below shoulders to above shoulders
        - Face / legs NOT required
        """

        if landmarks is None:
            return None

        lm = self.landmarks_to_array(landmarks, image_shape)

        required = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_WRIST', 'RIGHT_WRIST'
        ]

        for r in required:
            if r not in lm or lm[r][2] < visibility_threshold:
                return None

        left_sh = np.array(lm['LEFT_SHOULDER'][:2])
        right_sh = np.array(lm['RIGHT_SHOULDER'][:2])
        left_wr = np.array(lm['LEFT_WRIST'][:2])
        right_wr = np.array(lm['RIGHT_WRIST'][:2])

        shoulder_mid_y = (left_sh[1] + right_sh[1]) / 2
        wrist_mid_y = (left_wr[1] + right_wr[1]) / 2

        torso_height = abs(left_sh[1] - lm['LEFT_HIP'][1]) if 'LEFT_HIP' in lm else 200
        if torso_height < 1e-6:
            return None

        # wrist relative position
        rel = (shoulder_mid_y - wrist_mid_y) / torso_height

        is_up = rel > up_threshold
        is_down = rel < down_threshold

        return {
            "is_up": is_up,
            "is_down": is_down,
            "relative_height": rel
        }

    def detect_standing_leg_raise(
        self,
        landmarks,
        image_shape,
        min_lift_ratio: float = 0.05,   # 🔥 bahut loose (5% lift bhi chalega)
        visibility_threshold: float = 0.2
    ):
        """
        ULTRA-LOOSE Standing Leg Raise detection
        - Sirf leg ka thoda sa upar jaana kaafi
        - Dusra paav, hip, balance -> IGNORE
        - Face / hands -> IGNORE
        """

        if landmarks is None:
            return None

        lm = self.landmarks_to_array(landmarks, image_shape)

        # bas itna dikhe ki legs exist kar rahi hain
        required = [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]

        for r in required:
            if r not in lm or lm[r][2] < visibility_threshold:
                return None

        left_hip = np.array(lm['LEFT_HIP'][:2])
        right_hip = np.array(lm['RIGHT_HIP'][:2])

        # reference height (hips distance)
        ref = np.linalg.norm(left_hip - right_hip)
        if ref < 1e-6:
            return None

        def get_leg_y(ankle, knee):
            if lm[ankle][2] >= visibility_threshold:
                return lm[ankle][1]
            return lm[knee][1]

        left_leg_y = get_leg_y('LEFT_ANKLE', 'LEFT_KNEE')
        right_leg_y = get_leg_y('RIGHT_ANKLE', 'RIGHT_KNEE')

        # y axis downwards → upar = negative
        left_lift = (left_hip[1] - left_leg_y) / ref
        right_lift = (right_hip[1] - right_leg_y) / ref

        # 🔥 koi bhi ek leg thoda upar → VALID
        if left_lift > min_lift_ratio:
            return {
                "side": "left",
                "lift_ratio": left_lift
            }

        if right_lift > min_lift_ratio:
            return {
                "side": "right",
                "lift_ratio": right_lift
            }

        return None

    def close(self):
        """Release resources held by MediaPipe Pose instance."""
        # guard in case initializer failed
        if hasattr(self, "pose") and self.pose is not None:
            self.pose.close()
