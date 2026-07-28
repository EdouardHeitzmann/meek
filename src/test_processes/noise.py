from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import warnings

import numpy as np
from numpy.typing import NDArray


SENTINEL = np.int16(-127)


class ImplicitSampler:
    def __init__(
        self,
        noise_level: float,
        n_cands: int,
        wt_vec: NDArray[np.floating],
        sample_size: int | None = None,
        fractional_sample_size: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
        cache_rows: bool = False,
        aggressive: bool = True,
    ) -> None:
        self.noise_level = float(noise_level)
        self.n_cands = int(n_cands)
        self.with_replacement = bool(with_replacement)
        self.seed = self._seed_from_melbourne_date() if seed is None else int(seed)
        self.cache_rows = bool(cache_rows)
        self.aggressive = bool(aggressive)
        self._row_cache: dict[tuple[int, int, int], NDArray[np.integer]] = {}

        weights = self._validate_weights(wt_vec)
        total_ballot_wt = int(weights.sum())
        self.sample_size = self._resolve_sample_size(
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            total_ballot_wt=total_ballot_wt,
        )
        self._validate_sampling_scale(total_ballot_wt)

        rng = np.random.default_rng(self.seed)
        sampled_serials = self._sample_serials(
            rng=rng,
            total_ballot_wt=total_ballot_wt,
        )
        self.sampled_serials = sampled_serials
        self.sampled_indices = self._serials_to_row_indices(
            sampled_serials=sampled_serials,
            weights=weights,
        )
        self.noised_mask = self._make_noised_mask(rng, sampled_serials)
        self.target_indices = self._make_target_indices(rng, sampled_serials)
        self.noise_keys = np.zeros(self.sample_size, dtype=np.uint64)
        if np.any(self.noised_mask):
            self.noise_keys[self.noised_mask] = self.hash_keys(
                sampled_serials[self.noised_mask],
            )

    @classmethod
    def from_profile(
        cls,
        profile,
        noise_level: float,
        n_cands: int | None = None,
        **kwargs,
    ) -> ImplicitSampler:
        wt_vec = cls._wt_vec_from_profile(profile)
        if n_cands is None:
            if not hasattr(profile, "candidates"):
                raise ValueError("n_cands is required when profile lacks candidates.")
            n_cands = len(profile.candidates)
        return cls(
            noise_level=noise_level,
            n_cands=n_cands,
            wt_vec=wt_vec,
            **kwargs,
        )

    @staticmethod
    def _wt_vec_from_profile(profile) -> NDArray[np.float64]:
        if hasattr(profile, "wt_vec"):
            return np.asarray(profile.wt_vec, dtype=np.float64)
        if hasattr(profile, "df") and "Weight" in profile.df:
            return profile.df["Weight"].astype(np.float64).to_numpy()
        raise ValueError("profile must expose wt_vec or df['Weight'].")

    def sample(
        self,
        i: int,
        ballot_matrix: NDArray[np.integer],
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        if i < 0 or i >= self.sample_size:
            raise IndexError(f"sample index {i} outside [0, {self.sample_size}).")

        row_idx = int(self.sampled_indices[i])
        row_c = ballot_matrix[row_idx]

        if not self.noised_mask[i]:
            return row_c, row_c

        key = int(self.noise_keys[i])
        target_index = int(self.target_indices[i])
        cache_key = (row_idx, key, target_index)
        if self.cache_rows and cache_key in self._row_cache:
            return row_c, self._row_cache[cache_key]

        row_b = self._noise_row(row_c, key, target_index=target_index)
        if self.cache_rows:
            self._row_cache[cache_key] = row_b
        return row_c, row_b

    def hash_keys(self, values: NDArray[np.integer]) -> NDArray[np.uint64]:
        x = np.asarray(values, dtype=np.uint64)
        x = x + np.uint64(self.seed) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    def _validate_weights(self, wt_vec: NDArray[np.floating]) -> NDArray[np.int64]:
        weights = np.asarray(wt_vec, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("wt_vec must be one-dimensional.")
        if np.any(weights < 0):
            raise ValueError("wt_vec cannot contain negative weights.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("wt_vec must contain finite values.")
        if not np.allclose(weights, np.round(weights)):
            raise ValueError("wt_vec must contain integer-valued weights.")

        int_weights = weights.astype(np.int64)
        if int_weights.sum() <= 0:
            raise ValueError("wt_vec must have positive total weight.")
        return int_weights

    def _resolve_sample_size(
        self,
        sample_size: int | None,
        fractional_sample_size: float | None,
        total_ballot_wt: int,
    ) -> int:
        has_sample_size = sample_size is not None
        has_fraction = fractional_sample_size is not None
        if has_sample_size == has_fraction:
            raise ValueError(
                "Exactly one of sample_size or fractional_sample_size is required."
            )

        if has_sample_size:
            resolved = int(sample_size)
            if resolved < 0:
                raise ValueError("sample_size must be non-negative.")
            return resolved

        fraction = float(fractional_sample_size)
        if fraction < 0.0:
            raise ValueError("fractional_sample_size must be non-negative.")
        return int(total_ballot_wt * fraction)

    def _validate_sampling_scale(self, total_ballot_wt: int) -> None:
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("noise_level must be in [0, 1].")
        if self.n_cands <= 0:
            raise ValueError("n_cands must be positive.")

        if self.sample_size >= total_ballot_wt:
            message = (
                f"sample_size={self.sample_size} is not smaller than total "
                f"ballot weight {total_ballot_wt}."
            )
            if self.with_replacement:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            else:
                raise ValueError(message)

    def _sample_serials(
        self,
        rng: np.random.Generator,
        total_ballot_wt: int,
    ) -> NDArray[np.int64]:
        if self.sample_size == 0:
            return np.array([], dtype=np.int64)

        if self.with_replacement:
            return rng.integers(
                0,
                total_ballot_wt,
                size=self.sample_size,
                dtype=np.int64,
            )

        return rng.choice(
            total_ballot_wt,
            size=self.sample_size,
            replace=False,
        ).astype(np.int64, copy=False)

    def _serials_to_row_indices(
        self,
        sampled_serials: NDArray[np.int64],
        weights: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        if len(sampled_serials) == 0:
            return np.array([], dtype=np.int64)

        cumulative = np.cumsum(weights)
        return np.searchsorted(cumulative, sampled_serials, side="right").astype(
            np.int64,
            copy=False,
        )

    def _make_noised_mask(
        self,
        rng: np.random.Generator,
        sampled_serials: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        mask = np.zeros(self.sample_size, dtype=np.bool_)
        n_noised = int(round(self.noise_level * self.sample_size))
        if n_noised == 0:
            return mask

        selected_positions = rng.choice(
            self.sample_size,
            size=n_noised,
            replace=False,
        )

        if self.with_replacement:
            noised_serials = np.unique(sampled_serials[selected_positions])
            return np.isin(sampled_serials, noised_serials)

        mask[selected_positions] = True
        return mask

    def _make_target_indices(
        self,
        rng: np.random.Generator,
        sampled_serials: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        targets = self._draw_target_indices(rng, self.sample_size)
        if not self.with_replacement or self.sample_size == 0:
            return targets

        normalized = targets.copy()
        first_by_serial: dict[int, int] = {}
        for position, serial in enumerate(sampled_serials):
            serial_int = int(serial)
            target = first_by_serial.setdefault(serial_int, int(targets[position]))
            normalized[position] = target
        return normalized

    def _draw_target_indices(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.int64]:
        if size == 0:
            return np.array([], dtype=np.int64)
        scale = max(float(self.n_cands + 1), 1.0) / np.log(10.0)
        return np.floor(rng.exponential(scale=scale, size=size)).astype(np.int64)

    def _noise_row(
        self,
        row: NDArray[np.integer],
        key: int,
        *,
        target_index: int | None = None,
    ) -> NDArray[np.integer]:
        rng = np.random.default_rng(key)
        operation_order = self._operation_order(rng)
        target = None if not self.aggressive else target_index

        for operation in operation_order:
            noised = self._apply_noise_operation(row, operation, rng, target)
            if noised is not None and not np.array_equal(noised, row):
                return noised

        raise ValueError("Unable to noise selected row.")

    def _operation_order(self, rng: np.random.Generator) -> list[int]:
        first = int(rng.integers(0, 5))
        return [first] + [op for op in range(5) if op != first]

    def _apply_noise_operation(
        self,
        row: NDArray[np.integer],
        operation: int,
        rng: np.random.Generator,
        target_index: int | None = None,
    ) -> NDArray[np.integer] | None:
        if operation == 0:
            return self._replace_candidate(row, rng, target_index)
        if operation == 1:
            return self._insert_candidate(row, rng, target_index)
        if operation == 2:
            return self._swap_candidates(row, rng, target_index)
        if operation == 3:
            return self._delete_candidate(row, rng, target_index)
        if operation == 4:
            return self._append_candidates(row, rng)
        raise ValueError(f"Unknown noising operation: {operation}.")

    def _replace_candidate(
        self,
        row: NDArray[np.integer],
        rng: np.random.Generator,
        target_index: int | None = None,
    ) -> NDArray[np.integer] | None:
        valid_positions = self._valid_positions(row)
        missing = self._missing_candidates(row)
        if len(valid_positions) == 0 or len(missing) == 0:
            return None

        noised = np.array(row, copy=True)
        position = int(self._targeted_position(valid_positions, target_index, rng))
        noised[position] = int(rng.choice(missing))
        return noised

    def _insert_candidate(
        self,
        row: NDArray[np.integer],
        rng: np.random.Generator,
        target_index: int | None = None,
    ) -> NDArray[np.integer] | None:
        ranking = self._ranking_values(row)
        missing = self._missing_candidates(row)
        if len(missing) == 0 or len(ranking) >= len(row):
            return None

        insert_at = self._targeted_offset(len(ranking) + 1, target_index, rng)
        inserted = int(rng.choice(missing))
        new_ranking = np.insert(ranking, insert_at, inserted)
        return self._pad_ranking(new_ranking, len(row), row.dtype)

    def _swap_candidates(
        self,
        row: NDArray[np.integer],
        rng: np.random.Generator,
        target_index: int | None = None,
    ) -> NDArray[np.integer] | None:
        valid_positions = self._valid_positions(row)
        if len(valid_positions) < 2:
            return None

        noised = np.array(row, copy=True)
        i_idx = self._targeted_offset(len(valid_positions), target_index, rng)
        if target_index is None:
            j_idx = int(rng.integers(0, len(valid_positions) - 1))
        else:
            secondary_target = int(self._draw_target_indices(rng, 1)[0])
            j_idx = self._targeted_offset(len(valid_positions), secondary_target, rng)
            if j_idx == i_idx:
                j_idx = (j_idx + 1) % len(valid_positions)
        if target_index is None and j_idx >= i_idx:
            j_idx += 1
        i = valid_positions[i_idx]
        j = valid_positions[j_idx]
        noised[i], noised[j] = noised[j], noised[i]
        return noised

    def _delete_candidate(
        self,
        row: NDArray[np.integer],
        rng: np.random.Generator,
        target_index: int | None = None,
    ) -> NDArray[np.integer] | None:
        ranking = self._ranking_values(row)
        if len(ranking) == 0:
            return None

        delete_at = self._targeted_offset(len(ranking), target_index, rng)
        new_ranking = np.delete(ranking, delete_at)
        return self._pad_ranking(new_ranking, len(row), row.dtype)

    def _targeted_position(
        self,
        positions: NDArray[np.int64],
        target_index: int | None,
        rng: np.random.Generator,
    ) -> int:
        offset = self._targeted_offset(len(positions), target_index, rng)
        return int(positions[offset])

    def _targeted_offset(
        self,
        length: int,
        target_index: int | None,
        rng: np.random.Generator,
    ) -> int:
        if length <= 0:
            raise ValueError("length must be positive.")
        if target_index is None:
            return int(rng.integers(0, length))
        return min(max(int(target_index), 0), length - 1)

    def _append_candidates(
        self,
        row: NDArray[np.integer],
        rng: np.random.Generator,
    ) -> NDArray[np.integer] | None:
        ranking = self._ranking_values(row)
        missing = self._missing_candidates(row)
        trailing_slots = len(row) - len(ranking)
        if trailing_slots <= 0 or len(missing) == 0:
            return None

        n_to_append = int(rng.integers(1, min(trailing_slots, len(missing)) + 1))
        appended = rng.choice(missing, size=n_to_append, replace=False)
        new_ranking = np.concatenate((ranking, appended.astype(ranking.dtype)))
        return self._pad_ranking(new_ranking, len(row), row.dtype)

    def _valid_positions(self, row: NDArray[np.integer]) -> NDArray[np.int64]:
        return np.flatnonzero(np.asarray(row) >= 0).astype(np.int64, copy=False)

    def _ranking_values(self, row: NDArray[np.integer]) -> NDArray[np.integer]:
        return np.asarray(row)[np.asarray(row) >= 0]

    def _missing_candidates(self, row: NDArray[np.integer]) -> NDArray[np.int64]:
        present = set(int(candidate) for candidate in self._ranking_values(row))
        return np.asarray(
            [
                candidate
                for candidate in range(self.n_cands)
                if candidate not in present
            ],
            dtype=np.int64,
        )

    def _pad_ranking(
        self,
        ranking: NDArray[np.integer],
        width: int,
        dtype: np.dtype,
    ) -> NDArray[np.integer]:
        padded = np.full(width, SENTINEL, dtype=dtype)
        clipped = ranking[:width]
        padded[: len(clipped)] = clipped
        return padded

    def _seed_from_melbourne_date(self) -> int:
        today = datetime.now(ZoneInfo("Australia/Melbourne")).date()
        return int(today.strftime("%Y%m%d"))


__all__ = ["ImplicitSampler"]
