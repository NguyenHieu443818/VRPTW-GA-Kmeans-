"""Independent route-candidate generation for the FSRD-SP pipeline.

This module intentionally only creates and deduplicates a route pool.  It does
not select a global solution; that responsibility belongs to the later Set
Partitioning phase.  Keeping it separate makes GA and Discrete PSO directly
comparable as route generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from algorithm.DiscretePSO import DiscretePSO
from algorithm.GeneticAlgorithm import GA
from ultility.vrptw_evaluator import evaluate_route


@dataclass(frozen=True)
class RouteCandidate:
    """A capacity/depot-feasible route recovered from a generator."""

    customers: tuple[int, ...]
    fitness: float
    distance: float
    source: str


class RoutePool:
    """Deduplicated collection of candidate routes keyed by customer order."""

    def __init__(self) -> None:
        self._routes: dict[tuple[int, ...], RouteCandidate] = {}

    def add(self, candidate: RouteCandidate) -> bool:
        """Add a route, retaining the lower-fitness copy when it is duplicated."""
        current = self._routes.get(candidate.customers)
        if current is None or candidate.fitness < current.fitness:
            self._routes[candidate.customers] = candidate
            return True
        return False

    @property
    def routes(self) -> list[RouteCandidate]:
        return list(self._routes.values())

    def __len__(self) -> int:
        return len(self._routes)

    def coverage_count(self, customer: int) -> int:
        return sum(customer in route.customers for route in self._routes.values())


class FSRDRouteGenerator:
    """Run GA/PSO independently on overlapping subproblems and collect routes."""

    def __init__(
        self,
        *,
        customers: np.ndarray,
        graph_data: np.ndarray,
        vehicle_capacity: float,
        time_tolerance: float = 0.0,
        ga_params: Mapping[str, float | int] | None = None,
        pso_params: Mapping[str, float | int] | None = None,
    ) -> None:
        self.customers = customers
        self.graph_data = graph_data
        self.vehicle_capacity = vehicle_capacity
        self.time_tolerance = time_tolerance
        self.ga_params = dict(ga_params or {})
        self.pso_params = dict(pso_params or {})
        self.process_time = 0.0

    def _candidate(self, route: Sequence[int], source: str) -> RouteCandidate:
        customer_ids = tuple(int(customer) for customer in route)
        fitness, distance = evaluate_route(
            np.asarray(customer_ids, dtype=np.int64),
            self.customers,
            self.graph_data,
            self.vehicle_capacity,
            self.time_tolerance,
        )
        return RouteCandidate(customer_ids, float(fitness), float(distance), source)

    @staticmethod
    def _extract_routes(best_route_global: Iterable) -> Iterable[Sequence[int]]:
        """Flatten the route nesting returned by the existing GA/PSO classes."""
        for cluster_routes in best_route_global:
            for route in cluster_routes:
                if len(route) > 0:
                    yield route

    def _run_ga(self, cluster: list[int]) -> Iterable[Sequence[int]]:
        solver = GA(
            individual=int(self.ga_params.get("individual", 100)),
            generation=int(self.ga_params.get("generation", 100)),
            crossover_rate=float(self.ga_params.get("crossover_rate", 0.8)),
            mutation_rate=float(self.ga_params.get("mutation_rate", 0.15)),
            vehicle_capacity=self.vehicle_capacity,
            conserve_rate=float(self.ga_params.get("conserve_rate", 0.1)),
            M=self.time_tolerance,
            customers=self.customers,
            graph_data=self.graph_data,
        )
        _, routes, _, _, _ = solver.fit(clusters=[cluster])
        return self._extract_routes(routes)

    def _run_pso(self, cluster: list[int]) -> Iterable[Sequence[int]]:
        solver = DiscretePSO(
            num_particles=int(self.pso_params.get("num_particles", 30)),
            max_iter=int(self.pso_params.get("max_iter", 100)),
            vehicle_capacity=self.vehicle_capacity,
            M=self.time_tolerance,
            w=float(self.pso_params.get("w", 0.8)),
            c1=float(self.pso_params.get("c1", 0.5)),
            c2=float(self.pso_params.get("c2", 0.5)),
            customers=self.customers,
            graph_data=self.graph_data,
            sditer=int(self.pso_params.get("sditer", 50)),
        )
        _, routes, _, _, _ = solver.fit(clusters=[cluster])
        return self._extract_routes(routes)

    def generate(
        self,
        clusters: Iterable[Sequence[int]],
        methods: Sequence[str] = ("ga",),
    ) -> RoutePool:
        """Generate a route pool from all non-empty overlapping subproblems.

        Supported methods are ``"ga"`` and ``"pso"``.  The method deliberately
        returns a pool rather than a final VRPTW solution: overlapping clusters
        require Set Partitioning to select routes without duplicate coverage.
        """
        normalized_methods = tuple(method.lower() for method in methods)
        unsupported = set(normalized_methods) - {"ga", "pso"}
        if unsupported:
            raise ValueError(f"Unsupported route generators: {sorted(unsupported)}")

        start = perf_counter()
        pool = RoutePool()
        for cluster in clusters:
            customer_cluster = [int(customer) for customer in cluster]
            if not customer_cluster:
                continue
            for method in normalized_methods:
                routes = self._run_ga(customer_cluster) if method == "ga" else self._run_pso(customer_cluster)
                for route in routes:
                    pool.add(self._candidate(route, method.upper()))

        self.process_time = perf_counter() - start
        return pool
