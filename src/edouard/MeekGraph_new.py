import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from .ranking_election.meek import MeekCore
from .utils import convert_pf_to_numpy_arrays


class MeekGraph:
    """Streaming Meek graph builder with resumable disk persistence."""

    def __init__(
        self,
        profile,
        m,
        auditable_margins_per_deg,
        autosave_path,
        use_numerical_labels=False,
        autosave_layers=True,
        autosave_interval=None,
        resume=True,
        show_progress=True,
        save_layers_json=True,
        full_graph_path=None,
        safe_mode=True,
    ):
        self.profile = profile
        self.ballot_matrix, self.mult_vec, self.fpv_vec = convert_pf_to_numpy_arrays(profile)
        self.m = m
        self.num_cands = len(profile.candidates)
        self.auditable_margins_per_deg = auditable_margins_per_deg
        self.candidates = list(profile.candidates)
        self.use_numerical_labels = use_numerical_labels

        self.autosave_path = Path(autosave_path)
        self.autosave_layers = autosave_layers
        self.autosave_interval = autosave_interval
        self.resume = resume
        self.show_progress = show_progress
        self.save_layers_json = save_layers_json
        self.full_graph_path = Path(full_graph_path) if full_graph_path else None
        self.safe_mode = safe_mode

        self.stream_dir = self._stream_dir()
        self.layers_dir = self.stream_dir / "layers"
        self.edges_dir = self.stream_dir / "edges"
        self.index_dir = self.stream_dir / "indexes"
        self.meta_dir = self.stream_dir / "meta"

        self._ensure_dirs()
        self._initialize_or_resume()
        self._build_stream()

    def _stream_dir(self):
        base = self.autosave_path
        if base.suffix:
            base = base.with_suffix("")
        return Path(f"{base}_stream")

    def _ensure_dirs(self):
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.edges_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _layer_jsonl_path(self, layer):
        return self.layers_dir / f"layer_{layer}.jsonl"

    def _layer_json_path(self, layer):
        return self.layers_dir / f"layer_{layer}.json"

    def _edges_jsonl_path(self, layer):
        return self.edges_dir / f"edges_{layer}.jsonl"

    def _meta_jsonl_path(self, layer):
        return self.meta_dir / f"meta_{layer}.jsonl"

    def _index_db_path(self, layer):
        return self.index_dir / f"layer_{layer}.sqlite"

    def _index_to_letters(self, idx):
        letters = ""
        n = idx
        while True:
            n, rem = divmod(n, 26)
            letters = chr(ord("A") + rem) + letters
            if n == 0:
                break
            n -= 1
        return letters

    def _letters_to_index(self, letters):
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    def _node_id(self, layer, position):
        return f"{layer}{self._index_to_letters(position)}"

    def _split_node_id(self, node_id):
        digits = ""
        letters = ""
        for ch in node_id:
            if ch.isdigit():
                digits += ch
            else:
                letters += ch
        if not digits or not letters:
            raise ValueError(f"Invalid node id: {node_id}")
        return int(digits), self._letters_to_index(letters)

    def _node_to_string(self, winners, losers):
        winners = sorted(winners)
        losers = sorted(losers)
        return f"W:{','.join(map(str, winners))}_L:{','.join(map(str, losers))}"

    def _write_jsonl(self, path, record):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _iter_jsonl(self, path, start_index=0):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < start_index:
                    continue
                yield idx, json.loads(line)

    def _count_jsonl(self, path):
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _open_index(self, layer):
        path = self._index_db_path(layer)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS node_index (node_str TEXT PRIMARY KEY, node_id TEXT NOT NULL)"
        )
        return conn

    def _index_lookup(self, conn, node_str):
        cur = conn.execute("SELECT node_id FROM node_index WHERE node_str = ?", (node_str,))
        row = cur.fetchone()
        return row[0] if row else None

    def _index_insert(self, conn, node_str, node_id):
        conn.execute("INSERT OR IGNORE INTO node_index (node_str, node_id) VALUES (?, ?)", (node_str, node_id))

    def _save_state(self, state):
        payload = {
            "metadata": {
                "m": self.m,
                "num_cands": self.num_cands,
                "use_numerical_labels": self.use_numerical_labels,
                "candidates": self.candidates,
            },
            "stream_state": state,
        }
        if self.autosave_path.exists() and self.safe_mode:
            backup = self.autosave_path.with_suffix(self.autosave_path.suffix + ".bak")
            if not backup.exists():
                backup.write_bytes(self.autosave_path.read_bytes())
        self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
        with self.autosave_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_state(self):
        if not self.autosave_path.exists():
            return None
        with self.autosave_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metadata = payload.get("metadata", {})
        if metadata:
            if metadata.get("m") != self.m:
                raise ValueError("Saved m does not match current m.")
            if metadata.get("num_cands") != self.num_cands:
                raise ValueError("Saved num_cands does not match current num_cands.")
        return payload.get("stream_state")

    def _is_old_format(self):
        if not self.autosave_path.exists():
            return False
        with self.autosave_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return "nodes" in payload and "layers" in payload

    def _stream_ready(self):
        layer0 = self._layer_jsonl_path(0)
        return layer0.exists() and layer0.stat().st_size > 0

    def _initialize_or_resume(self):
        if self.autosave_path.exists() and not self.resume:
            raise RuntimeError(
                f"Autosave file exists at {self.autosave_path}; refusing to overwrite without resume=True."
            )

        if self.resume and self.autosave_path.exists():
            if self._is_old_format():
                self._migrate_old_json()
            else:
                state = self._load_state()
                if state is None:
                    raise RuntimeError(
                        "Autosave file exists but has no stream_state and is not old-format."
                    )
                if not self._stream_ready():
                    raise RuntimeError(
                        "Stream state exists but streaming files are missing. Refusing to overwrite."
                    )
                return

        if not self.autosave_path.exists():
            root_node = {
                "node_id": self._node_id(0, 0),
                "node_str": self._node_to_string([], []),
                "winner_to_cand": [],
                "initial_losers": [],
                "layer": 0,
                "position_in_layer": 0,
            }
            self._write_jsonl(self._layer_jsonl_path(0), root_node)
            state = {
                "current_layer": 0,
                "current_index": 0,
                "next_layer_count": 0,
                "build_complete": False,
                "root_id": root_node["node_id"],
            }
            self._save_state(state)

    def _migrate_old_json(self):
        if self._stream_ready() and self.safe_mode:
            raise RuntimeError("Stream storage already exists; refusing to overwrite during migration.")

        with self.autosave_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        backup = self.autosave_path.with_suffix(self.autosave_path.suffix + ".bak")
        if self.safe_mode and not backup.exists():
            backup.write_bytes(self.autosave_path.read_bytes())

        layers = data.get("layers", [])
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])

        for layer_idx, layer_nodes in enumerate(layers):
            for node_id in layer_nodes:
                node_record = nodes.get(node_id)
                if node_record:
                    self._write_jsonl(self._layer_jsonl_path(layer_idx), node_record)

        for edge in edges:
            from_id = edge.get("from")
            if from_id is None:
                continue
            from_record = nodes.get(from_id)
            layer_idx = from_record.get("layer", 0) if from_record else 0
            self._write_jsonl(self._edges_jsonl_path(layer_idx), edge)

        for layer_idx, layer_nodes in enumerate(layers):
            conn = self._open_index(layer_idx)
            for node_id in layer_nodes:
                node_record = nodes.get(node_id)
                node_str = node_record.get("node_str") if node_record else None
                if node_str:
                    self._index_insert(conn, node_str, node_id)
            conn.commit()
            conn.close()

        state = {
            "current_layer": len(layers) - 1 if layers else 0,
            "current_index": 0,
            "next_layer_count": 0,
            "build_complete": bool(data.get("build_complete", False)),
            "root_id": data.get("root_id", "0A"),
        }
        self._save_state(state)

    def _edge_label(self, from_node, to_node):
        from_winners = set(from_node["winner_to_cand"])
        to_winners = set(to_node["winner_to_cand"])
        from_losers = set(from_node["initial_losers"])
        to_losers = set(to_node["initial_losers"])

        new_winners = to_winners - from_winners
        new_losers = to_losers - from_losers

        if new_winners:
            candidate_idx = list(new_winners)[0]
            if self.use_numerical_labels:
                return f"Elect {candidate_idx}"
            return f"Elect {self.candidates[candidate_idx]}"
        if new_losers:
            candidate_idx = list(new_losers)[0]
            if self.use_numerical_labels:
                return f"Eliminate {candidate_idx}"
            return f"Eliminate {self.candidates[candidate_idx]}"
        return ""

    def _expand_node(self, node):
        winners = node["winner_to_cand"]
        losers = node["initial_losers"]

        if len(winners) >= self.m:
            return [], {}

        try:
            core = MeekCore(
                profile=self.profile,
                m=self.m,
                candidates=list(self.profile.candidates),
                winner_to_cand=winners,
                initial_losers=losers,
                tiebreak="first_place",
            )
            tallies, helper_vecs, play, tiebreak, quota = core._run_first_round()
        except Exception as exc:
            print(f"Error processing node {node.get('node_id', '?')}: {exc}")
            return [], {}

        deg = len(winners)
        margin = self.auditable_margins_per_deg[deg]

        active_candidates = set(range(self.num_cands)) - set(winners) - set(losers)
        active_tallies = tallies[list(active_candidates)] if active_candidates else []

        meta = {
            "canonical_loser": None,
            "canonical_winner": None,
            "other_plausible_winners": None,
            "rejected_winners": None,
            "probable_winner": None,
        }

        if len(active_tallies) > 0:
            lowest = np.min(active_tallies)
            canonical = int(np.argwhere(tallies == lowest).flatten()[0])
            meta["canonical_loser"] = canonical

        tallies_masked = tallies.copy()
        if winners:
            tallies_masked[winners] = -1

        possible_nodes = []

        if np.any(tallies_masked > quota + margin) and len(winners) != 2:
            only_winner = int(np.argmax(tallies_masked))
            meta["canonical_winner"] = only_winner
            max_tally = tallies_masked[only_winner]
            within = np.where(tallies_masked >= max_tally - margin)[0]
            other_plausible = []
            possible_nodes.append({
                "winner_to_cand": winners + [only_winner],
                "initial_losers": losers,
            })
            for cand in within:
                cand = int(cand)
                if cand != only_winner:
                    other_plausible.append(cand)
                    possible_nodes.append({
                        "winner_to_cand": winners + [cand],
                        "initial_losers": losers,
                    })
            meta["other_plausible_winners"] = other_plausible
        else:
            if np.any(tallies_masked > quota - margin):
                possible_mask = tallies_masked > quota - margin
                biggest = int(np.argmax(tallies_masked))
                within = np.where(
                    possible_mask & (tallies_masked >= tallies_masked[biggest] - margin)
                )[0]
                for cand in within:
                    cand = int(cand)
                    possible_nodes.append({
                        "winner_to_cand": winners + [cand],
                        "initial_losers": losers,
                    })
                rejected = np.where(
                    possible_mask & (tallies_masked < tallies_masked[biggest] - margin)
                )[0]
                if len(rejected) > 0:
                    meta["rejected_winners"] = [int(i) for i in rejected]
                    meta["probable_winner"] = biggest

            allow_eliminations = len(active_candidates) + len(winners) != self.m
            if active_candidates and allow_eliminations:
                lowest = np.min(active_tallies) if len(active_tallies) > 0 else None
                if lowest is not None:
                    within = np.where(tallies <= lowest + margin)[0]
                    for cand in within:
                        cand = int(cand)
                        if cand in active_candidates:
                            possible_nodes.append({
                                "winner_to_cand": winners,
                                "initial_losers": losers + [cand],
                            })

        return possible_nodes, meta

    def _build_stream(self):
        state = self._load_state()
        if state is None:
            return
        if state.get("build_complete"):
            return

        total_layers = max(self.num_cands - self.m, 0)
        while True:
            state = self._load_state()
            if state is None:
                break
            if state.get("build_complete"):
                break

            current_layer = state["current_layer"]
            current_index = state["current_index"]
            next_layer_count = state["next_layer_count"]

            layer_path = self._layer_jsonl_path(current_layer)
            if not layer_path.exists():
                raise RuntimeError(f"Missing layer file: {layer_path}")

            if self.show_progress:
                print(f"Layer {current_layer}/{total_layers} (start index {current_index})")

            next_layer = current_layer + 1
            conn = self._open_index(next_layer)
            pending = 0

            processed = 0
            total_nodes = self._count_jsonl(layer_path)
            for idx, node in self._iter_jsonl(layer_path, start_index=current_index):
                possible_nodes, meta = self._expand_node(node)
                meta_record = {"node_id": node["node_id"], **meta}
                self._write_jsonl(self._meta_jsonl_path(current_layer), meta_record)

                for possible in possible_nodes:
                    node_str = self._node_to_string(
                        possible["winner_to_cand"],
                        possible["initial_losers"],
                    )
                    existing_id = self._index_lookup(conn, node_str)
                    if existing_id:
                        child_id = existing_id
                    else:
                        child_id = self._node_id(next_layer, next_layer_count)
                        next_layer_count += 1
                        self._index_insert(conn, node_str, child_id)
                        record = {
                            "node_id": child_id,
                            "node_str": node_str,
                            "winner_to_cand": list(possible["winner_to_cand"]),
                            "initial_losers": list(possible["initial_losers"]),
                            "layer": next_layer,
                            "position_in_layer": next_layer_count - 1,
                        }
                        self._write_jsonl(self._layer_jsonl_path(next_layer), record)

                    edge = {
                        "from": node["node_id"],
                        "to": child_id,
                        "label": self._edge_label(node, possible),
                    }
                    self._write_jsonl(self._edges_jsonl_path(current_layer), edge)

                processed += 1
                pending += 1

                if self.autosave_interval and processed % self.autosave_interval == 0:
                    state = {
                        "current_layer": current_layer,
                        "current_index": idx + 1,
                        "next_layer_count": next_layer_count,
                        "build_complete": False,
                        "root_id": state.get("root_id", "0A"),
                    }
                    self._save_state(state)
                    conn.commit()
                    pending = 0

            if pending:
                conn.commit()
            conn.close()

            if self.save_layers_json:
                self._write_layer_json(current_layer)

            if next_layer_count == 0 or current_layer >= self.num_cands:
                final_state = {
                    "current_layer": current_layer,
                    "current_index": 0,
                    "next_layer_count": 0,
                    "build_complete": True,
                    "root_id": state.get("root_id", "0A"),
                }
                self._save_state(final_state)
                if self.full_graph_path:
                    self.save_full_graph_json(self.full_graph_path)
                break

            state = {
                "current_layer": next_layer,
                "current_index": 0,
                "next_layer_count": 0,
                "build_complete": False,
                "root_id": state.get("root_id", "0A"),
            }
            if self.autosave_layers:
                self._save_state(state)

    def _write_layer_json(self, layer):
        json_path = self._layer_json_path(layer)
        if json_path.exists() and self.safe_mode:
            return
        tmp_path = json_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write("[")
            first = True
            for _, record in self._iter_jsonl(self._layer_jsonl_path(layer)):
                if not first:
                    f.write(",")
                f.write(json.dumps(record))
                first = False
            f.write("]")
        tmp_path.replace(json_path)

    def save_full_graph_json(self, path):
        target = Path(path)
        if target.exists() and self.safe_mode:
            backup = target.with_suffix(target.suffix + ".bak")
            if not backup.exists():
                backup.write_bytes(target.read_bytes())
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("{\n")
            f.write("\"metadata\":")
            f.write(json.dumps({
                "m": self.m,
                "num_cands": self.num_cands,
                "use_numerical_labels": self.use_numerical_labels,
                "candidates": self.candidates,
            }))
            f.write(",\n\"nodes\":[")
            first = True
            layer = 0
            while True:
                path = self._layer_jsonl_path(layer)
                if not path.exists():
                    break
                for _, record in self._iter_jsonl(path):
                    if not first:
                        f.write(",")
                    f.write(json.dumps(record))
                    first = False
                layer += 1
            f.write("],\n\"edges\":[")
            first = True
            layer = 0
            while True:
                path = self._edges_jsonl_path(layer)
                if not path.exists():
                    break
                for _, record in self._iter_jsonl(path):
                    if not first:
                        f.write(",")
                    f.write(json.dumps(record))
                    first = False
                layer += 1
            f.write("]\n}")
        tmp.replace(target)

    def lookup_node(self, node_id):
        layer, position = self._split_node_id(node_id)
        node = None
        for idx, record in self._iter_jsonl(self._layer_jsonl_path(layer), start_index=position):
            if idx == position:
                node = record
                break
        if node is None:
            raise ValueError(f"Node {node_id} not found.")

        meta = {}
        meta_path = self._meta_jsonl_path(layer)
        for _, record in self._iter_jsonl(meta_path):
            if record.get("node_id") == node_id:
                meta = record
                break

        winners = [self.candidates[i] for i in node["winner_to_cand"]]
        losers = [self.candidates[i] for i in node["initial_losers"]]
        return {
            "node_id": node_id,
            "layer": node["layer"],
            "position_in_layer": node["position_in_layer"],
            "winner_to_cand": node["winner_to_cand"],
            "initial_losers": node["initial_losers"],
            "winners": winners,
            "losers": losers,
            "canonical_loser": meta.get("canonical_loser"),
            "canonical_winner": meta.get("canonical_winner"),
            "other_plausible_winners": meta.get("other_plausible_winners"),
            "rejected_winners": meta.get("rejected_winners"),
            "probable_winner": meta.get("probable_winner"),
        }
