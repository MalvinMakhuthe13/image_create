from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from app.engine.matching.models import ImageFeatures


@dataclass(frozen=True)
class PairPose:
    left: int
    right: int
    rotation: np.ndarray
    translation: np.ndarray
    weight: float


def _components(image_count: int, edges: list[PairPose]) -> list[list[int]]:
    adjacency = [[] for _ in range(image_count)]
    for edge in edges:
        adjacency[edge.left].append(edge.right)
        adjacency[edge.right].append(edge.left)
    seen, groups = set(), []
    for start in range(image_count):
        if start in seen:
            continue
        queue, group = deque([start]), []
        seen.add(start)
        while queue:
            node = queue.popleft()
            group.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        groups.append(sorted(group))
    return groups


def _initial_camera_poses(image_count: int, edges: list[PairPose]):
    """Propagate relative world-to-camera transforms along a maximum-confidence tree."""
    adjacency = [[] for _ in range(image_count)]
    for edge in edges:
        adjacency[edge.left].append((edge.right, edge, True))
        adjacency[edge.right].append((edge.left, edge, False))
    rotations, translations, visited = {}, {}, set()
    groups = _components(image_count, edges)
    for group in groups:
        root = group[0]
        rotations[root] = np.eye(3)
        translations[root] = np.zeros(3)
        visited.add(root)
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour, edge, forward in sorted(
                adjacency[current], key=lambda item: -item[1].weight
            ):
                if neighbour in visited:
                    continue
                if forward:
                    relative_r, relative_t = edge.rotation, edge.translation
                else:
                    relative_r = edge.rotation.T
                    relative_t = -relative_r @ edge.translation
                rotations[neighbour] = relative_r @ rotations[current]
                translations[neighbour] = (
                    relative_r @ translations[current] + relative_t
                )
                visited.add(neighbour)
                queue.append(neighbour)
    return rotations, translations, groups


def _camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert a world-to-camera extrinsic transform to its world-space camera centre."""
    return -rotation.T @ translation


def _build_tracks(pair_rows, features):
    """Union keypoint observations connected by verified pairwise inliers."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in pair_rows:
        i, j = row["image_indices"]
        matches = row["_matches"]
        for match_index in row["_inliers"]:
            match = matches[match_index]
            union((i, match.query_index), (j, match.train_index))

    tracks = defaultdict(list)
    for image_index, keypoint_index in list(parent):
        tracks[find((image_index, keypoint_index))].append((image_index, keypoint_index))
    valid_tracks = []
    for rows in tracks.values():
        image_ids = [image_id for image_id, _ in rows]
        # Conflicting tracks with two detections in one photograph are rejected,
        # not silently collapsed by overwriting one keypoint.
        if len(rows) >= 2 and len(image_ids) == len(set(image_ids)):
            valid_tracks.append(dict(rows))
    return valid_tracks


def reconstruct_sparse_scene(features: list[ImageFeatures], pair_rows: list[dict],
                             *, max_tracks: int = 500, max_nfev: int = 40) -> dict:
    """Build and robustly refine a calibrated sparse multi-view reconstruction.

    This is a sparse SfM foundation. It does not infer dense surfaces or coverage.
    Each disconnected component has its own arbitrary scale and coordinate frame.
    """
    edges = []
    usable_rows = []
    for row in pair_rows:
        pose = row.get("pose")
        if not pose or pose.get("status") != "estimated":
            continue
        i, j = row["image_indices"]
        if features[i].camera_matrix is None or features[j].camera_matrix is None:
            continue
        edges.append(PairPose(
            i, j,
            np.asarray(pose["rotation_matrix_left_to_right"], dtype=float),
            np.asarray(pose["translation_direction_left_to_right"], dtype=float),
            max(1.0, float(pose.get("pose_inlier_count", 0))),
        ))
        usable_rows.append(row)

    if not edges:
        return {
            "status": "unavailable",
            "reason": "no_calibrated_relative_pose_edges",
            "camera_count": len(features),
            "components": [],
            "coverage_assessment": {"status": "not_assessed"},
        }

    rotations, translations, groups = _initial_camera_poses(len(features), edges)
    tracks = _build_tracks(usable_rows, features)
    output_components = []
    for component in groups:
        component_edges = [edge for edge in edges if edge.left in component and edge.right in component]
        if len(component) < 2 or not component_edges:
            output_components.append({
                "image_indices": component,
                "status": "insufficient_camera_edges",
                "camera_count": len(component),
                "landmark_count": 0,
            })
            continue

        component_tracks = [
            track for track in tracks
            if len(set(track).intersection(component)) >= 2
        ]
        component_tracks.sort(key=lambda track: (-len(track), tuple(sorted(track.items()))))
        component_tracks = component_tracks[:max_tracks]

        # Triangulate each track from its first two available observations.
        points, observations = [], []
        for track in component_tracks:
            image_ids = sorted(set(track).intersection(component))
            if len(image_ids) < 2:
                continue
            a, b = image_ids[:2]
            fa, fb = features[a], features[b]
            ka, kb = track[a], track[b]
            pa = np.asarray(fa.keypoints[ka], dtype=float)
            pb = np.asarray(fb.keypoints[kb], dtype=float)
            na = cv2.undistortPoints(pa.reshape(1, 1, 2), fa.camera_matrix, fa.distortion_coefficients).reshape(2)
            nb = cv2.undistortPoints(pb.reshape(1, 1, 2), fb.camera_matrix, fb.distortion_coefficients).reshape(2)
            projection_a = np.hstack((rotations[a], translations[a].reshape(3, 1)))
            projection_b = np.hstack((rotations[b], translations[b].reshape(3, 1)))
            homogeneous = cv2.triangulatePoints(
                projection_a, projection_b, na.reshape(2, 1), nb.reshape(2, 1)
            ).reshape(4)
            if abs(homogeneous[3]) < 1e-10:
                continue
            point = homogeneous[:3] / homogeneous[3]
            if not np.isfinite(point).all():
                continue
            point_observations = []
            for image_id in image_ids:
                keypoint_id = track[image_id]
                point_observations.append((
                    image_id,
                    np.asarray(features[image_id].keypoints[keypoint_id], dtype=float),
                ))
            points.append(point)
            observations.append(point_observations)

        if len(points) < 8:
            output_components.append({
                "image_indices": component,
                "status": "insufficient_multiview_tracks",
                "camera_count": len(component),
                "landmark_count": len(points),
                "track_count": len(component_tracks),
            })
            continue

        # Fix the first camera and one baseline to remove global gauge freedoms.
        camera_ids = component
        root = camera_ids[0]
        variable_cameras = camera_ids[1:]
        initial_rvecs = {
            idx: Rotation.from_matrix(rotations[idx]).as_rotvec()
            for idx in camera_ids
        }
        initial_t = {idx: translations[idx].copy() for idx in camera_ids}
        baseline_edge = component_edges[0]
        baseline_length = 1.0
        initial_baseline = np.linalg.norm(
            _camera_center(rotations[baseline_edge.right], initial_t[baseline_edge.right])
            - _camera_center(rotations[baseline_edge.left], initial_t[baseline_edge.left])
        )
        if initial_baseline > 1e-8:
            baseline_length = initial_baseline

        point_array = np.asarray(points, dtype=float)
        camera_offset = {idx: 6 * offset for offset, idx in enumerate(variable_cameras)}
        point_offset = 6 * len(variable_cameras)

        x0 = []
        for idx in variable_cameras:
            x0.extend(initial_rvecs[idx])
            x0.extend(initial_t[idx])
        x0.extend(point_array.ravel())

        flat_observations = []
        for point_id, point_obs in enumerate(observations):
            for image_id, pixel in point_obs:
                flat_observations.append((point_id, image_id, pixel))

        def unpack(parameters):
            camera_r, camera_t = {root: np.eye(3)}, {root: np.zeros(3)}
            for idx in variable_cameras:
                offset = camera_offset[idx]
                camera_r[idx] = Rotation.from_rotvec(parameters[offset:offset + 3]).as_matrix()
                camera_t[idx] = parameters[offset + 3:offset + 6]
            point_values = parameters[point_offset:].reshape(-1, 3)
            return camera_r, camera_t, point_values

        jacobian_pattern = lil_matrix(
            (2 * len(flat_observations) + 1, len(x0)), dtype=np.int8
        )
        for row_index, (point_id, image_id, _) in enumerate(flat_observations):
            if image_id != root:
                camera_start = camera_offset[image_id]
                jacobian_pattern[2 * row_index:2 * row_index + 2,
                                 camera_start:camera_start + 6] = 1
            point_start = point_offset + point_id * 3
            jacobian_pattern[2 * row_index:2 * row_index + 2,
                             point_start:point_start + 3] = 1
        # Camera-centre distance depends on both rotation and translation for
        # world-to-camera extrinsics, so include all six endpoint parameters.
        for endpoint in (baseline_edge.left, baseline_edge.right):
            if endpoint != root:
                camera_start = camera_offset[endpoint]
                jacobian_pattern[-1, camera_start:camera_start + 6] = 1

        def residual(parameters):
            camera_r, camera_t, point_values = unpack(parameters)
            values = []
            for point_id, image_id, pixel in flat_observations:
                point_camera = camera_r[image_id] @ point_values[point_id] + camera_t[image_id]
                if point_camera[2] <= 1e-6:
                    values.extend([1000.0, 1000.0])
                    continue
                projected, _ = cv2.projectPoints(
                    point_camera.reshape(1, 1, 3),
                    np.zeros(3), np.zeros(3),
                    features[image_id].camera_matrix,
                    features[image_id].distortion_coefficients,
                )
                projected = projected.reshape(2)
                values.extend(projected - pixel)
            # Soft scale gauge: preserve camera-centre distance, not the
            # difference between world-to-camera translation vectors. The latter
            # is not a world-space baseline when camera rotations differ.
            left_center = _camera_center(
                camera_r[baseline_edge.left], camera_t[baseline_edge.left]
            )
            right_center = _camera_center(
                camera_r[baseline_edge.right], camera_t[baseline_edge.right]
            )
            values.append((np.linalg.norm(right_center - left_center) - baseline_length) * 10.0)
            return np.asarray(values, dtype=float)

        try:
            optimized = least_squares(
                residual, np.asarray(x0), loss="soft_l1", f_scale=2.0,
                max_nfev=max_nfev, verbose=0,
                jac_sparsity=jacobian_pattern.tocsr(), tr_solver="lsmr",
            )
            camera_r, camera_t, optimized_points = unpack(optimized.x)
            residual_values = residual(optimized.x)
            reprojection_values = residual_values[:-1].reshape(-1, 2)
            median_error = float(np.median(np.linalg.norm(reprojection_values, axis=1)))
            output_components.append({
                "image_indices": component,
                "status": "optimized" if optimized.success else "optimization_stopped",
                "camera_count": len(component),
                "landmark_count": int(len(optimized_points)),
                "multi_view_track_count": int(len(observations)),
                "observation_count": len(flat_observations),
                "median_reprojection_error_px": round(median_error, 4),
                "optimizer": "scipy_least_squares_soft_l1",
                "optimizer_success": bool(optimized.success),
                "optimizer_message": str(optimized.message),
                "coordinate_scale": "arbitrary_component_monocular_scale",
                "cameras": {
                    str(idx): {
                        "rotation_world_to_camera": np.round(camera_r[idx], 7).tolist(),
                        "translation_world_to_camera": np.round(camera_t[idx], 7).tolist(),
                        "intrinsics_source": features[idx].intrinsics_source,
                    } for idx in component
                },
                "sparse_points": np.round(optimized_points, 6).tolist(),
                "coverage_claim": "sparse_landmarks_only_not_dense_surface_coverage",
            })
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            output_components.append({
                "image_indices": component,
                "status": "optimization_failed",
                "camera_count": len(component),
                "landmark_count": len(points),
                "reason": str(exc)[:300],
            })

    return {
        "status": "complete" if output_components else "unavailable",
        "method": "incremental_pairwise_initialization_then_sparse_global_bundle_adjustment",
        "camera_count": len(features),
        "calibrated_pair_edges": len(edges),
        "component_count": len(groups),
        "components": output_components,
        "scale": "each_component_has_arbitrary_monocular_scale",
        "coverage_assessment": {
            "status": "not_assessed",
            "reason": "Sparse landmarks do not measure dense visible or occluded surface coverage.",
        },
        "limitations": [
            "Intrinsics may be approximate EXIF-derived focal estimates.",
            "No lens distortion calibration is included.",
            "Disconnected components have independent arbitrary coordinate frames and scales.",
            "Sparse feature tracks do not provide dense surfaces, watertight meshes or occlusion maps.",
            "This optimizer is a bounded CPU prototype, not a substitute for a mature SfM system on large datasets.",
        ],
    }
