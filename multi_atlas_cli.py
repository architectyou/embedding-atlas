#!/usr/bin/env python3
"""
Multi-Dataset Embedding Atlas CLI
각 parquet 파일별로 별도의 endpoint를 제공하는 embedding atlas 서버
"""

import logging
import socket
from pathlib import Path
from typing import Dict

import click
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from packages.backend.embedding_atlas.data_source import DataSource
from packages.backend.embedding_atlas.server import make_server
from packages.backend.embedding_atlas.utils import Hasher


class MultiDatasetAtlas:
    """다중 데이터셋을 관리하는 Embedding Atlas"""

    def __init__(self, static_path: str, duckdb_uri: str = "wasm"):
        self.datasets: Dict[str, DataSource] = {}
        self.static_path = static_path
        self.duckdb_uri = duckdb_uri
        self.app = FastAPI(title="Multi-Dataset Embedding Atlas")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        self._setup_routes()

    def add_dataset(self, name: str, parquet_path: str, vector_column: str = "vector"):
        print(f"Loading dataset '{name}' from {parquet_path}")

        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        id_column = "_row_index"
        if id_column not in df.columns:
            df[id_column] = range(df.shape[0])

        text_column = None
        for col in ["text", "content", "description", "sentence", "document"]:
            if col in df.columns:
                text_column = col
                break

        metadata = {
            "columns": {
                "id": id_column,
                "text": text_column,
            },
            "dataset_name": name,
            "file_path": parquet_path,
            "vector_column": vector_column,
        }

        if vector_column in df.columns:
            x_col, y_col = self._compute_or_find_coordinates(df, vector_column)
            if x_col and y_col:
                metadata["columns"]["embedding"] = {"x": x_col, "y": y_col}

        neighbors_column = None
        for col in ["__neighbors", "neighbors", "neighbor_ids"]:
            if col in df.columns:
                neighbors_column = col
                break

        if neighbors_column:
            metadata["columns"]["neighbors"] = neighbors_column
            print(f"Found neighbors column: {neighbors_column}")

        hasher = Hasher()
        hasher.update([parquet_path])
        hasher.update(metadata)
        identifier = hasher.hexdigest()

        dataset = DataSource(identifier, df, metadata)
        self.datasets[name] = dataset

        print(f"Dataset '{name}' added successfully")
        return dataset

    def _compute_or_find_coordinates(self, df, vector_column):
        x_candidates = ["x", "projection_x", "umap_x", "tsne_x", "embedding_x"]
        y_candidates = ["y", "projection_y", "umap_y", "tsne_y", "embedding_y"]

        x_col = None
        y_col = None

        for col in x_candidates:
            if col in df.columns:
                x_col = col
                break

        for col in y_candidates:
            if col in df.columns:
                y_col = col
                break

        if x_col and y_col:
            return x_col, y_col

        if vector_column in df.columns:
            print(f"Computing UMAP projection for {vector_column}...")
            try:
                import numpy as np
                import umap
                from sklearn.neighbors import NearestNeighbors

                vectors = []
                for _, row in df.iterrows():
                    vector = row[vector_column]
                    if isinstance(vector, (list, np.ndarray)):
                        vectors.append(np.array(vector))
                    else:
                        vectors.append(vector)

                vectors = np.array(vectors)

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )

                projection = reducer.fit_transform(vectors)
                x_col = "projection_x"
                y_col = "projection_y"
                df[x_col] = projection[:, 0]
                df[y_col] = projection[:, 1]

                print("Computing nearest neighbors for similarity search...")
                nn = NearestNeighbors(n_neighbors=21, metric="cosine")
                nn.fit(vectors)

                neighbors_list = []
                for i in range(len(vectors)):
                    distances, indices = nn.kneighbors([vectors[i]])
                    neighbor_indices = [int(idx) for idx in indices[0][1:21]]
                    neighbor_distances = [float(dist) for dist in distances[0][1:21]]
                    neighbors_list.append(
                        {"distances": neighbor_distances, "ids": neighbor_indices}
                    )

                df["__neighbors"] = neighbors_list

                print(f"UMAP projection completed: {x_col}, {y_col}")
                print("Nearest neighbors computed for similarity search")
                return x_col, y_col

            except ImportError:
                print("UMAP or sklearn not available, skipping projection")
                return None, None
            except Exception as e:
                print(f"Error computing projection: {e}")
                return None, None

        return None, None

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "message": "Multi-Dataset Embedding Atlas",
                "datasets": list(self.datasets.keys()),
                "endpoints": [f"/dataset/{name}" for name in self.datasets.keys()],
            }

        @self.app.get("/datasets")
        async def list_datasets():
            result = {}
            for name, dataset in self.datasets.items():
                metadata = dataset.metadata.copy()
                metadata.update(
                    {
                        "num_rows": len(dataset.dataset),
                        "num_columns": len(dataset.dataset.columns),
                        "columns": list(dataset.dataset.columns),
                    }
                )
                result[name] = metadata
            return result

        @self.app.get("/dataset/{dataset_name}")
        async def redirect_to_dataset(dataset_name: str):
            if dataset_name not in self.datasets:
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{dataset_name}' not found"
                )
            return RedirectResponse(url=f"/dataset/{dataset_name}/")

        @self.app.get("/dataset/{dataset_name}/info")
        async def get_dataset_info(dataset_name: str):
            if dataset_name not in self.datasets:
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{dataset_name}' not found"
                )

            dataset = self.datasets[dataset_name]
            metadata = dataset.metadata.copy()
            metadata.update(
                {
                    "num_rows": len(dataset.dataset),
                    "num_columns": len(dataset.dataset.columns),
                    "columns": list(dataset.dataset.columns),
                }
            )
            return metadata

    def mount_dataset_routes(self):
        for name, dataset in self.datasets.items():
            dataset_app = make_server(
                dataset, static_path=self.static_path, duckdb_uri=self.duckdb_uri
            )
            self.app.mount(f"/dataset/{name}", dataset_app)
            print(f"Mounted dataset '{name}' at /dataset/{name}")


def find_available_port(start_port: int, max_attempts: int = 10, host="localhost"):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("No available ports found in the given range")


@click.command()
@click.option("--host", default="localhost", help="Host address")
@click.option("--port", default=5055, help="Port number")
@click.option("--auto-port/--no-auto-port", default=True, help="Auto find available port")
@click.option("--duckdb", default="wasm", help="DuckDB connection mode")
@click.option("--static", default=None, help="Custom static files path")
@click.option(
    "--datasets-dir",
    default=None,
    help="Parquet directory to auto-load (default: <repo>/results/atlas/facets)",
)
@click.option(
    "--vector-column",
    default="embedding",
    help="Vector column name for auto-loaded parquet datasets",
)
def main(host, port, auto_port, duckdb, static, datasets_dir, vector_column):
    """Multi-Dataset Embedding Atlas CLI"""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = Path(__file__).parent

    if static is None:
        backend_path = repo_root / "packages" / "backend" / "embedding_atlas" / "static"
        static = str(backend_path.resolve())

    if datasets_dir is None:
        datasets_dir = str((repo_root / "results" / "atlas" / "facets").resolve())

    ds_dir = Path(datasets_dir).expanduser().resolve()

    print(f"Using static files from: {static}")
    print(f"Using datasets dir: {ds_dir}")

    atlas = MultiDatasetAtlas(static_path=static, duckdb_uri=duckdb)

    loaded = 0

    if ds_dir.exists() and ds_dir.is_dir():
        parquet_files = sorted(ds_dir.glob("*.parquet"))
        for file_path in parquet_files:
            dataset_name = file_path.stem
            try:
                atlas.add_dataset(dataset_name, str(file_path), vector_column)
                loaded += 1
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

    if loaded == 0:
        print("No datasets loaded from facets directory. Fallback to parquet_data presets...")
        parquet_data_dir = repo_root / "parquet_data"
        legacy_files = [
            ("scouting_report_openai_masked", "scouting_report_openai_masked.parquet", "vector"),
            ("scouting_report_openai_unmasked", "scouting_report_openai_with_year.parquet", "vector"),
            ("scouting_report_bgem3", "bgem3_masked.parquet", "vector"),
        ]
        for name, file_name, vector_col in legacy_files:
            file_path = parquet_data_dir / file_name
            if file_path.exists():
                try:
                    atlas.add_dataset(name, str(file_path), vector_col)
                    loaded += 1
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    if not atlas.datasets:
        print("No datasets loaded! Check --datasets-dir or parquet_data files.")
        return

    atlas.mount_dataset_routes()

    if auto_port:
        final_port = find_available_port(port, max_attempts=10, host=host)
        if final_port != port:
            logging.info(f"Port {port} not available, using {final_port}")
    else:
        final_port = port

    print("\n🗺️  Multi-Dataset Embedding Atlas")
    print(f"📊 Loaded {len(atlas.datasets)} datasets:")
    for name in atlas.datasets.keys():
        print(f"   • {name}: http://{host}:{final_port}/dataset/{name}")
    print(f"📋 All datasets: http://{host}:{final_port}/datasets")
    print(f"🌐 Server starting at http://{host}:{final_port}")

    uvicorn.run(atlas.app, host=host, port=final_port, access_log=False)


if __name__ == "__main__":
    main()
