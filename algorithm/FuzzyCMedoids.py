"""Fuzzy c-medoids decomposition for the VRPTW DRI framework.

This implementation follows Section 4.1.1, Section 4.1.2, and Algorithm 2
of Kerscher and Minner (2024). Its clustering distance is the paper's
spatial-temporal-demand (STD) distance, not Euclidean distance over a
normalised feature vector.
"""

from __future__ import annotations

import time

import numpy as np


# Input columns: [ID, X, Y, demand, ready_time, due_time, service_time]
C_X = 1
C_Y = 2
C_DEMAND = 3
C_READY_TIME = 4
C_DUE_TIME = 5
C_SERVICE_TIME = 6

# Internal feature columns: tau_i = (x, y, theta, e, l, s, d)
F_X = 0
F_Y = 1
F_THETA = 2
F_READY = 3
F_DUE = 4
F_SERVICE = 5
F_DEMAND = 6


class FuzzyCMedoids:
    """Fuzzy c-medoids using the DRI spatial-temporal-demand metric.

    ``vehicle_capacity`` is required because Eq. (13) includes the demand
    penalty ``(d_i + d_j) / Q``. ``spatial_angle_weight`` is lambda in Eq.
    (35); its default 1.0 follows the paper's Homberger benchmark setup.
    """

    def __init__(
        self,
        n_cluster: int = 5,
        vehicle_capacity: float | None = None,
        spatial_angle_weight: float = 1.0,
        kappa: float = 2.0,
        epsilon: float = 1e-4,
        max_iter: int = 300,
        rho: float = 0.3,
        seed: int = 42,
    ):
        if n_cluster < 1:
            raise ValueError("n_cluster must be at least 1.")
        if kappa <= 1:
            raise ValueError("kappa must be greater than 1.")
        if epsilon <= 0 or max_iter < 1:
            raise ValueError("epsilon and max_iter must be positive.")
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1].")
        if spatial_angle_weight < 0:
            raise ValueError("spatial_angle_weight must be non-negative.")
        if vehicle_capacity is not None and vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive.")

        self.n_cluster = n_cluster
        self.vehicle_capacity = vehicle_capacity
        self.spatial_angle_weight = spatial_angle_weight
        self.kappa = kappa
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.rho = rho
        self.seed = seed

        self.U: np.ndarray | None = None
        self.medoids: list[int] = []  # input customer indices, 1-indexed
        self.labels: np.ndarray | None = None
        self.process_time = 0.0
        self.n_iter = 0

        self._features: np.ndarray | None = None
        self._capacity: float | None = None
        self._operation_duration: float | None = None

    @staticmethod
    def _build_feature_matrix(customers: np.ndarray) -> np.ndarray:
        """Build ``tau_i = (x_i, y_i, theta_i, e_i, l_i, s_i, d_i)``."""
        depot = customers[0, C_X:C_Y + 1].astype(float)
        cust = customers[1:]
        theta = np.arctan2(
            cust[:, C_Y].astype(float) - depot[1],
            cust[:, C_X].astype(float) - depot[0],
        )
        return np.column_stack(
            (
                cust[:, C_X].astype(float),
                cust[:, C_Y].astype(float),
                theta,
                cust[:, C_READY_TIME].astype(float),
                cust[:, C_DUE_TIME].astype(float),
                cust[:, C_SERVICE_TIME].astype(float),
                cust[:, C_DEMAND].astype(float),
            )
        )

    def _directional_std_distance(
        self, origins: np.ndarray, destinations: np.ndarray
    ) -> np.ndarray:
        """Calculate directional STD distance of Eq. (13) for ``i -> j``.

        Solomon/Homberger instances use Euclidean distance as both travel cost
        and travel time. For an asymmetric/custom travel-time matrix, replace
        the ``travel_time`` calculation here with that matrix lookup.
        """
        if self._capacity is None or self._operation_duration is None:
            raise RuntimeError("Distance metric is unavailable before fit().")

        dx = destinations[None, :, F_X] - origins[:, None, F_X]
        dy = destinations[None, :, F_Y] - origins[:, None, F_Y]
        travel_time = np.hypot(dx, dy)
        spatial = np.sqrt(
            dx**2
            + dy**2
            + self.spatial_angle_weight
            * (destinations[None, :, F_THETA] - origins[:, None, F_THETA]) ** 2
        )

        # Eq. (11) f_ij and Eq. (12) h_ij.
        flexibility = destinations[None, :, F_DUE] - (
            origins[:, None, F_READY]
            + origins[:, None, F_SERVICE]
            + travel_time
        )
        waiting = np.maximum(
            destinations[None, :, F_READY]
            - (origins[:, None, F_DUE] + origins[:, None, F_SERVICE] + travel_time),
            0.0,
        )
        capacity_use = (
            origins[:, None, F_DEMAND] + destinations[None, :, F_DEMAND]
        ) / self._capacity

        return spatial * (
            2.0 - (flexibility - waiting) / self._operation_duration + capacity_use
        )

    def _std_distance_matrix(
        self, origins: np.ndarray, destinations: np.ndarray
    ) -> np.ndarray:
        """Calculate Eq. (14): ``min(S_std(i,j), S_std(j,i))``."""
        forward = self._directional_std_distance(origins, destinations)
        reverse = self._directional_std_distance(destinations, origins).T
        return np.minimum(forward, reverse)

    def _initialize_U(self, n_customers: int) -> np.ndarray:
        """Randomly initialise ``U^0`` as stated in Algorithm 2."""
        rng = np.random.default_rng(self.seed)
        U = rng.random((n_customers, self.n_cluster))
        return U / U.sum(axis=1, keepdims=True)

    def _compute_prototypes(self, features: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Calculate fuzzy weighted virtual feature vectors for q clusters.

        Normalised ``mu**kappa`` weights are consistent with objective (19) and
        preserve the units of the original feature vector.
        """
        weights = U**self.kappa
        return (weights.T @ features) / weights.sum(axis=0)[:, None]

    def _select_medoids(self, features: np.ndarray, prototypes: np.ndarray) -> list[int]:
        """Project virtual prototypes to distinct customer medoids via Eq. (14)."""
        distance = self._std_distance_matrix(features, prototypes)
        selected: list[tuple[int, int]] = []
        available = np.ones(features.shape[0], dtype=bool)

        # This greedy matching is deterministic and prevents duplicate medoids,
        # so every hard-assigned cluster contains at least its own medoid.
        for cluster in np.argsort(distance.min(axis=0)):
            candidates = np.argsort(distance[:, cluster])
            medoid = next(int(i) for i in candidates if available[i])
            selected.append((int(cluster), medoid))
            available[medoid] = False

        selected.sort(key=lambda pair: pair[0])
        return [medoid for _, medoid in selected]

    def _update_U(self, features: np.ndarray, medoid_indices: list[int]) -> np.ndarray:
        """Update memberships using Eq. (18), handling zero distance exactly."""
        distances = self._std_distance_matrix(features, features[medoid_indices])
        U = np.zeros_like(distances)
        zero_distance = np.isclose(distances, 0.0, atol=1e-12)
        has_zero = zero_distance.any(axis=1)

        # A medoid has membership 1 in its own cluster. Coincident locations
        # are assigned equally among their zero-distance medoids.
        U[has_zero] = zero_distance[has_zero] / zero_distance[has_zero].sum(
            axis=1, keepdims=True
        )
        nonzero = ~has_zero
        if np.any(nonzero):
            exponent = -2.0 / (self.kappa - 1.0)
            inverse_power = distances[nonzero] ** exponent
            U[nonzero] = inverse_power / inverse_power.sum(axis=1, keepdims=True)
        return U

    def fit(self, customers: np.ndarray, vehicle_capacity: float | None = None) -> tuple:
        """Fit Fuzzy c-medoids and return ``(U, medoids, labels, n_iter)``.

        The explicit ``vehicle_capacity`` argument overrides the constructor.
        It is mandatory to compute the demand penalty in Eq. (13).
        """
        customers = np.asarray(customers)
        if customers.ndim != 2 or customers.shape[1] < 7 or customers.shape[0] < 2:
            raise ValueError(
                "customers must contain one depot and at least one customer with 7 columns."
            )

        features = self._build_feature_matrix(customers)
        n_customers = features.shape[0]
        if self.n_cluster > n_customers:
            raise ValueError("n_cluster cannot exceed the number of customers.")

        capacity = vehicle_capacity if vehicle_capacity is not None else self.vehicle_capacity
        if capacity is None or capacity <= 0:
            raise ValueError("vehicle_capacity is required and must be positive.")
        duration = float(customers[0, C_DUE_TIME] - customers[0, C_READY_TIME])
        if duration <= 0:
            raise ValueError("Depot due_time must be greater than ready_time.")

        start = time.time()
        self._features = features
        self._capacity = float(capacity)
        self._operation_duration = duration
        U = self._initialize_U(n_customers)

        medoid_indices: list[int] = []
        for iteration in range(1, self.max_iter + 1):
            previous_U = U.copy()
            prototypes = self._compute_prototypes(features, U)
            medoid_indices = self._select_medoids(features, prototypes)
            U = self._update_U(features, medoid_indices)
            if np.linalg.norm(U - previous_U, ord="fro") < self.epsilon:
                break

        self.process_time = time.time() - start
        self.n_iter = iteration
        self.U = U
        self.medoids = [index + 1 for index in medoid_indices]
        self.labels = np.argmax(U, axis=1)
        # Resolve membership ties in favour of the corresponding medoid. This
        # preserves the paper's argmax assignment while ensuring that each
        # distinct medoid represents a non-empty hard subproblem.
        self.labels[medoid_indices] = np.arange(self.n_cluster)
        return U, self.medoids, self.labels, self.n_iter

    def get_clusters(self) -> list[list[int]]:
        """Hard-assign each customer to the maximum-membership cluster."""
        if self.labels is None:
            raise RuntimeError("Chưa chạy fit(). Hãy gọi fit(customers) trước.")
        return [
            (np.where(self.labels == cluster)[0] + 1).tolist()
            for cluster in range(self.n_cluster)
        ]

    def get_overlapping_clusters(
        self,
        rho: float | None = None,
        size_limit: int | None = None,
    ) -> list[list[int]]:
        """Create the rho-overlapping subproblems used by FSRD-SP.

        Every customer belongs to its primary cluster, defined by its maximum
        membership. A boundary customer (primary membership <= rho) is also
        placed in exactly one secondary cluster: the cluster with its second
        highest membership. Hence a customer occurs in one or two subproblems.
        ``size_limit`` may discard secondary members only; core members are
        never discarded.
        """
        if self.U is None:
            raise RuntimeError("Chưa chạy fit(). Hãy gọi fit(customers) trước.")
        if rho is None:
            rho = self.rho
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1].")
        if size_limit is not None and size_limit < 1:
            raise ValueError("size_limit must be positive when provided.")

        primary = np.argmax(self.U, axis=1)
        core_members = [
            (np.where(primary == cluster)[0] + 1).tolist()
            for cluster in range(self.n_cluster)
        ]
        boundary_members: list[list[tuple[int, float]]] = [
            [] for _ in range(self.n_cluster)
        ]

        if self.n_cluster > 1:
            for customer_index, primary_cluster in enumerate(primary):
                if self.U[customer_index, primary_cluster] > rho:
                    continue
                membership = self.U[customer_index].copy()
                membership[primary_cluster] = -np.inf
                secondary_cluster = int(np.argmax(membership))
                boundary_members[secondary_cluster].append(
                    (customer_index + 1, float(membership[secondary_cluster]))
                )

        clusters: list[list[int]] = []
        for cluster in range(self.n_cluster):
            secondary = sorted(
                boundary_members[cluster], key=lambda item: item[1], reverse=True
            )
            if size_limit is not None:
                slots = max(size_limit - len(core_members[cluster]), 0)
                secondary = secondary[:slots]
            clusters.append(core_members[cluster] + [customer for customer, _ in secondary])
        return clusters

    def get_overlapping_subproblems(
        self,
        rho: float | None = None,
        size_limit: int | None = None,
    ) -> list[list[int]]:
        """Alias for get_overlapping_clusters (FSRD-SP terminology)."""
        return self.get_overlapping_clusters(rho=rho, size_limit=size_limit)

    def get_fuzzy_boundary_customers(self) -> list[int]:
        """Return assigned customers with ``mu_i,p* <= rho`` for DRI LS."""
        if self.U is None:
            raise RuntimeError("Chưa chạy fit(). Hãy gọi fit(customers) trước.")
        assigned_membership = self.U.max(axis=1)
        return (np.where(assigned_membership <= self.rho)[0] + 1).tolist()

