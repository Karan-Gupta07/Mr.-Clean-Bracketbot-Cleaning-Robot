"""Download public FlyWire v783 tables and build a disclosed neuron subgraph.

Data: https://codex.flywire.ai/ and https://doi.org/10.5281/zenodo.10676866
Large source files and derived arrays stay under ignored out/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


BASE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"
FILES = ("neurons.csv.gz", "classification.csv.gz", "coordinates.csv.gz", "connections.csv.gz")
PINNED_SHA256 = {
    "neurons.csv.gz": "6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719",
    "classification.csv.gz": "e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663",
    "coordinates.csv.gz": "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05",
    "connections.csv.gz": "d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "sources.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for name in FILES:
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = directory / (name + ".part")
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(path)
        checksum = sha256(path)
        if checksum != PINNED_SHA256[name]:
            raise ValueError(f"{name} differs from the pinned v783 source; review the release before changing its checksum")
        if name in previous and checksum != previous[name]["sha256"]:
            raise ValueError(f"Checksum changed for {name}; inspect source changes before rebuilding")
        manifest[name] = {"url": BASE + name, "sha256": checksum, "bytes": path.stat().st_size}
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(directory, count):
    """Top-strength induced subset with sensory/descending interface quotas.

    This selection is a computational pilot, not an identified biological circuit.
    Use count=0 to retain all source neurons and edges.
    """
    neurons = pd.read_csv(directory / "neurons.csv.gz").set_index("root_id")
    classes = pd.read_csv(directory / "classification.csv.gz").set_index("root_id")
    neurons = neurons.join(classes).fillna("")
    print("Reading measured edges...", flush=True)
    edges = pd.read_csv(directory / "connections.csv.gz", usecols=["pre_root_id", "post_root_id", "syn_count"])
    edges = edges.groupby(["pre_root_id", "post_root_id"], sort=True, as_index=False).syn_count.sum()
    strength = edges.groupby("pre_root_id").syn_count.sum().add(edges.groupby("post_root_id").syn_count.sum(), fill_value=0)
    if count:
        if count < 32:
            raise ValueError("--neurons must be 0 (all), or at least 32")
        quota = count // 8
        candidates = neurons.assign(strength=strength).fillna({"strength": 0})
        def pick(mask, n):
            return candidates[mask].sort_values(["strength"], ascending=False, kind="stable").head(n).index.tolist()
        input_ids = pick(candidates.flow == "afferent", quota)
        output_ids = pick(candidates.super_class == "descending", quota)
        internal_ids = pick(candidates.super_class.isin(["central", "visual_projection"]), count - len(input_ids) - len(output_ids))
        ids = np.array(sorted(set(input_ids + output_ids + internal_ids)), dtype=np.int64)
    else:
        ids = np.array(sorted(neurons.index), dtype=np.int64)
    selected = neurons.loc[ids]
    edges = edges[edges.pre_root_id.isin(ids) & edges.post_root_id.isin(ids)]
    pre = np.searchsorted(ids, edges.pre_root_id.to_numpy())
    post = np.searchsorted(ids, edges.post_root_id.to_numpy())
    counts = edges.syn_count.to_numpy(dtype=np.float32)
    # Explicit engineering approximation: GABA/GLUT negative; ACH positive;
    # modulatory/unknown transmitters positive in this pilot, not a biological claim.
    signs = np.where(selected.nt_type.isin(["GABA", "GLUT"]), -1.0, 1.0).astype(np.float32)
    adjacency = sparse.coo_matrix((counts * signs[pre], (post, pre)), shape=(len(ids), len(ids))).tocsr()
    rowsum = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    adjacency = sparse.diags(1 / np.maximum(rowsum, 1)) @ adjacency
    inputs = np.flatnonzero(selected.flow.to_numpy() == "afferent")
    outputs = np.flatnonzero(selected.super_class.to_numpy() == "descending")
    reachable = np.zeros(len(ids), dtype=bool)
    reachable[inputs] = True
    for _ in range(len(ids)):
        nxt = reachable | (abs(adjacency) @ reachable > 0)
        if np.array_equal(nxt, reachable):
            break
        reachable = nxt
    outputs = outputs[reachable[outputs]]
    if not len(inputs) or not len(outputs):
        raise ValueError("Selected graph has no afferent-to-descending path; increase --neurons")
    # Source positions are annotated points on neurons, not necessarily somata.
    coordinates = pd.read_csv(directory / "coordinates.csv.gz")
    coordinates = coordinates[coordinates.root_id.isin(ids)].drop_duplicates("root_id").set_index("root_id")
    positions = np.full((len(ids), 3), np.nan, dtype=np.float32)
    for i, root_id in enumerate(ids):
        if root_id in coordinates.index:
            positions[i] = np.fromstring(coordinates.loc[root_id, "position"].strip("[]"), sep=" ")
    valid = np.isfinite(positions).all(axis=1)
    if not valid.all():
        raise ValueError(f"Missing anatomical positions for {(~valid).sum()} neurons")
    name = "full" if count == 0 else str(count)
    target = directory / f"graph_{name}.npz"
    coo = adjacency.tocoo()
    np.savez_compressed(target, ids=ids, pre=coo.col, post=coo.row, weight=coo.data.astype(np.float32),
                        positions=positions, inputs=inputs, outputs=outputs,
                        nt_type=selected.nt_type.to_numpy(dtype=str),
                        super_class=selected.super_class.to_numpy(dtype=str))
    manifest = {
        "dataset": "FlyWire FAFB v783", "neurons": len(ids), "edges": len(coo.data),
        "source_neurons": len(neurons), "subset": bool(count),
        "selection": "Top incident synapse strength, 1/8 afferent, 1/8 descending, remainder central/visual projection" if count else "All source neurons",
        "inputs": len(inputs), "reachable_outputs": len(outputs), "input_reachable_neurons": int(reachable.sum()),
        "weak_components": int(connected_components(adjacency, directed=True, connection="weak", return_labels=False)),
        "graph_sha256": sha256(target), "source_files": json.loads((directory / "sources.json").read_text()),
        "position_type": "First annotated point per neuron from coordinates.csv.gz; source coordinate units",
        "weight_rule": "Synapse counts aggregated by directed neuron pair, signed by presynaptic transmitter, normalized by absolute incoming sum",
        "sign_assumption": "GABA/GLUT=-1; all other/unknown=+1 (simplified engineering model)",
        "citation": "Dorkenwald et al. 2024, doi:10.1038/s41586-024-07558-y",
        "data_terms": "Consult FlyWire source release terms; do not redistribute source data as original project code",
    }
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, indent=2), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("out/flywire"))
    parser.add_argument("--neurons", type=int, default=512, help="Pilot subset size; 0 keeps the complete dataset")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    download(args.directory)
    if not args.download_only:
        build(args.directory, args.neurons)


if __name__ == "__main__":
    main()
