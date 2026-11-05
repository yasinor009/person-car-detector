"""Calibrated ground-contact ranging and independent temporal range states.

Camera coordinates: x right, y down, z forward. Ground plane n.P = height_m,
with unit normal pointing down. Input images MUST already be undistorted with K.
Returns distance to estimated ground contact, not the object's 3-D centre.
"""
from dataclasses import dataclass
import math
import hashlib
import json
import numpy as np


@dataclass
class GroundCalibration:
    camera_matrix: list
    image_size: list
    normal: list
    height_m: float
    height_std_m: float
    angle_std_deg: float
    pixel_std: float
    calibration_id: str
    fixed_camera: bool
    planar_ground: bool
    systematic_relative_std: float

    def __post_init__(self):
        self.k = np.asarray(self.camera_matrix, dtype=float)
        self.n = np.asarray(self.normal, dtype=float)
        if (self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                or self.k[0, 0] <= 0 or self.k[1, 1] <= 0
                or not np.allclose(self.k[2], [0, 0, 1]) or abs(np.linalg.det(self.k)) < 1e-9):
            raise ValueError('Invalid camera matrix')
        if self.n.shape != (3,) or not np.isfinite(self.n).all() or not np.isclose(np.linalg.norm(self.n), 1, atol=1e-4):
            raise ValueError('Ground normal must be a finite unit vector')
        if len(self.image_size) != 2 or any(type(v) is not int or v <= 0 for v in self.image_size):
            raise ValueError('Invalid calibration image size')
        for name in ('height_m', 'height_std_m', 'angle_std_deg', 'pixel_std', 'systematic_relative_std'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip() or self.fixed_camera is not True or self.planar_ground is not True:
            raise ValueError('Ground ranging requires explicit fixed-camera and planar-ground assumptions')

    def signature(self):
        """Include numeric calibration, not only the operator's reusable label."""
        values = [self.k.tolist(), self.n.tolist(), self.image_size, self.height_m,
                  self.height_std_m, self.angle_std_deg, self.pixel_std,
                  self.systematic_relative_std, self.calibration_id]
        return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()

    def measure(self, detection, shape):
        result = dict(raw_distance_m=None, filtered_distance_m=None, distance_m=None,
                      uncertainty_std_m=None, valid=False, quality='invalid',
                      method='calibrated_ground_contact', quantity='euclidean_ground_contact_range',
                      calibration_status='configured_not_field_validated', calibration_id=self.calibration_id,
                      bbox_quality='unverified_contact', reason='invalid_box')
        h, w = shape[:2]
        # Do not guess whether a resolution change represents scaling or cropping.
        if [w, h] != self.image_size:
            return dict(result, reason='calibration_resolution_mismatch')
        try:
            box = np.asarray(detection.get('bbox'), dtype=float)
        except (TypeError, ValueError):
            return result
        if box.shape != (4,) or not np.isfinite(box).all():
            return result
        x, y, bw, bh = box
        if bw < 4 or bh < 8 or x <= 1 or y <= 1 or x+bw >= w-1 or y+bh >= h-1 or detection.get('truncated'):
            return dict(result, reason='small_or_truncated_box')
        if detection.get('occluded', False):
            return dict(result, reason='occluded_contact')
        # Bbox bottom is only a contact proxy. Unknown occlusion/pose remains a
        # systematic error; calibrated uncertainty is not a coverage guarantee.
        ray = np.linalg.solve(self.k, [x+bw/2, y+bh, 1.])
        unit = ray / np.linalg.norm(ray)
        denominator = float(self.n @ unit)
        angular = math.radians(self.angle_std_deg)
        if denominator <= max(1e-6, 3*angular):
            return dict(result, reason='near_horizon_or_above_ground')
        distance = self.height_m / denominator
        # Exact derivative of h*||ray||/(n.ray); works with skew and tilt too.
        length = np.linalg.norm(ray)
        dot = float(self.n @ ray)
        gradient = self.height_m*(ray/(length*dot)-length*self.n/dot**2)
        derivatives = gradient @ np.linalg.inv(self.k)[:, :2]
        random_std = max(.01, float(np.linalg.norm(derivatives)*self.pixel_std))
        systematic_std = math.sqrt((distance*self.height_std_m/self.height_m)**2
                                  + (distance*angular*math.sqrt(max(0, 1-denominator**2))/denominator)**2
                                  + (distance*self.systematic_relative_std)**2)
        total = math.hypot(random_std, systematic_std)
        valid = 0 < distance <= 100 and 1.96*total/distance <= .10
        return dict(result, raw_distance_m=float(distance), random_std_m=random_std,
                    systematic_std_m=systematic_std, uncertainty_std_m=total, valid=valid,
                    bbox_quality='bottom_center_proxy', quality='conditional' if valid else 'invalid',
                    reason='ok_conditional_on_ground_contact' if valid else 'range_or_uncertainty_limit')


class RangeBank:
    """[range, range velocity] per ID; systematic error never averages away."""
    def __init__(self, acceleration_std=3., max_gap_s=1.):
        if not all(math.isfinite(v) and v > 0 for v in (acceleration_std, max_gap_s)):
            raise ValueError('Invalid motion configuration')
        self.states = {}
        self.acceleration_std = acceleration_std
        self.max_gap_s = max_gap_s
        self.last_time = None
        self.context = None

    def update(self, detections, timestamp, calibration, shape):
        if not math.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        context = (calibration.signature() if calibration else None, tuple(shape[:2]))
        if context != self.context:
            self.states.clear()
            self.context = context
        keys = [(d.get('track_id'), d.get('class')) for d in detections
                if d.get('class') in ('person', 'car') and d.get('track_id') is not None]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate track observations in one frame')
        if self.last_time is not None and timestamp <= self.last_time:
            self.states.clear()
        self.last_time = timestamp
        self.states = {k: s for k, s in self.states.items() if timestamp-s['time'] <= self.max_gap_s}
        output = []
        for d in detections:
            if d['class'] not in ('person', 'car'):
                continue
            measurement = calibration.measure(d, shape) if calibration else dict(
                raw_distance_m=None, filtered_distance_m=None, distance_m=None,
                uncertainty_std_m=None, valid=False, quality='invalid',
                method='none', calibration_status='missing', reason='calibration_required')
            key = (d.get('track_id'), d['class'], calibration.calibration_id if calibration else None)
            if key[0] is None:
                measurement.update(valid=False, quality='invalid', reason='missing_track_id')
            if measurement['valid']:
                z = measurement['raw_distance_m']
                variance = measurement['random_std_m']**2
                old = self.states.get(key)
                if old is None:
                    state, covariance = np.array([z, 0.]), np.diag([variance, 100.])
                else:
                    dt = timestamp-old['time']
                    f = np.array([[1., dt], [0., 1.]])
                    g = np.array([dt*dt/2, dt])
                    state = f @ old['state']
                    covariance = f @ old['covariance'] @ f.T + np.outer(g, g)*self.acceleration_std**2
                    residual = z-state[0]
                    if residual**2 > 9*(covariance[0, 0]+variance):
                        measurement.update(valid=False, quality='invalid', reason='temporal_outlier')
                    else:
                        gain = covariance[:, 0]/(covariance[0, 0]+variance)
                        state = state+gain*residual
                        a = np.eye(2)-np.outer(gain, [1., 0.])
                        covariance = a @ covariance @ a.T + np.outer(gain, gain)*variance
                if measurement['valid']:
                    self.states[key] = dict(state=state, covariance=covariance, time=timestamp)
                    uncertainty = math.sqrt(covariance[0, 0]+measurement['systematic_std_m']**2)
                    measurement.update(filtered_distance_m=float(state[0]), distance_m=float(state[0]),
                                       uncertainty_std_m=uncertainty, range_velocity_mps=float(state[1]))
                    if not 0 < state[0] <= 100 or 1.96*uncertainty/state[0] > .10:
                        measurement.update(valid=False, distance_m=None, quality='invalid', reason='filtered_uncertainty_limit')
            output.append(dict(d, distance=measurement, distance_m=measurement['distance_m'],
                               tracking_status='observed'))
        return output
