#!/usr/bin/env python3
"""
Multi-Dataset Embedding Atlas CLI
각 parquet 파일별로 별도의 endpoint를 제공하는 embedding atlas 서버
"""

import logging
import pathlib
import socket
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd
import uvicorn

# FastAPI와 기본 라우팅
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# embedding-atlas backend 모듈 임포트
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

        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        self._setup_routes()

    def add_dataset(self, name: str, parquet_path: str, vector_column: str = "vector"):
        """데이터셋 추가"""
        print(f"Loading dataset '{name}' from {parquet_path}")

        # Parquet 파일 로드
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # 메타데이터 설정
        id_column = "_row_index"
        if id_column not in df.columns:
            df[id_column] = range(df.shape[0])

        # 텍스트 컬럼 찾기 (text, content, description 등)
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

        # 벡터가 있다면 임베딩 정보 추가
        if vector_column in df.columns:
            # 벡터에서 x, y 좌표 계산 (UMAP 또는 미리 계산된 좌표)
            x_col, y_col = self._compute_or_find_coordinates(df, vector_column)
            if x_col and y_col:
                metadata["columns"]["embedding"] = {
                    "x": x_col,
                    "y": y_col,
                }

        # neighbors 컬럼 찾기 (__neighbors, neighbors 등) - UMAP 계산 후에 체크
        neighbors_column = None
        for col in ["__neighbors", "neighbors", "neighbor_ids"]:
            if col in df.columns:
                neighbors_column = col
                break

        # neighbors 컬럼이 있다면 메타데이터에 추가
        if neighbors_column:
            metadata["columns"]["neighbors"] = neighbors_column
            print(f"Found neighbors column: {neighbors_column}")

        # 데이터소스 생성
        hasher = Hasher()
        hasher.update([parquet_path])
        hasher.update(metadata)
        identifier = hasher.hexdigest()

        dataset = DataSource(identifier, df, metadata)
        self.datasets[name] = dataset

        print(f"Dataset '{name}' added successfully")
        return dataset

    def _compute_or_find_coordinates(self, df, vector_column):
        """좌표 계산 또는 기존 좌표 찾기"""
        # 이미 x, y 좌표가 있는지 확인
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

        # 좌표가 없으면 벡터로부터 UMAP 계산
        if vector_column in df.columns:
            print(f"Computing UMAP projection for {vector_column}...")
            try:
                import numpy as np
                import umap
                from sklearn.neighbors import NearestNeighbors

                # 벡터 추출
                vectors = []
                for idx, row in df.iterrows():
                    vector = row[vector_column]
                    if isinstance(vector, (list, np.ndarray)):
                        vectors.append(np.array(vector))
                    else:
                        vectors.append(vector)

                vectors = np.array(vectors)

                # UMAP 적용
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )

                projection = reducer.fit_transform(vectors)

                # DataFrame에 추가
                x_col = "projection_x"
                y_col = "projection_y"
                df[x_col] = projection[:, 0]
                df[y_col] = projection[:, 1]

                # Nearest neighbors 계산 (유사 선수 기능을 위해)
                print("Computing nearest neighbors for similarity search...")
                nn = NearestNeighbors(n_neighbors=21, metric='cosine')  # 20개 + 자기 자신
                nn.fit(vectors)

                # 각 점에 대한 nearest neighbors 저장
                neighbors_list = []
                for i in range(len(vectors)):
                    distances, indices = nn.kneighbors([vectors[i]])
                    # 자기 자신 제외하고 상위 20개
                    neighbor_indices = [int(idx) for idx in indices[0][1:21]]
                    neighbor_distances = [float(dist) for dist in distances[0][1:21]]
                    # 기본 CLI와 같은 형식으로 저장: {"distances": [...], "ids": [...]}
                    neighbors_list.append({
                        "distances": neighbor_distances,
                        "ids": neighbor_indices
                    })

                # DataFrame에 neighbors 추가
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
        """API 라우트 설정"""

        @self.app.get("/")
        async def root():
            """루트: 사용 가능한 데이터셋 목록"""
            return {
                "message": "Multi-Dataset Embedding Atlas",
                "datasets": list(self.datasets.keys()),
                "endpoints": [f"/dataset/{name}" for name in self.datasets.keys()],
            }

        @self.app.get("/datasets")
        async def list_datasets():
            """데이터셋 목록과 메타데이터"""
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
            """특정 데이터셋으로 리다이렉트"""
            if dataset_name not in self.datasets:
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{dataset_name}' not found"
                )

            # 정적 파일 서빙을 위해 dataset-specific 경로로 리다이렉트
            return RedirectResponse(url=f"/dataset/{dataset_name}/")

        @self.app.get("/dataset/{dataset_name}/info")
        async def get_dataset_info(dataset_name: str):
            """데이터셋 정보"""
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
        """각 데이터셋별로 경로 마운트"""
        for name, dataset in self.datasets.items():
            # 각 데이터셋용 개별 FastAPI 앱 생성
            dataset_app = make_server(
                dataset, static_path=self.static_path, duckdb_uri=self.duckdb_uri
            )

            # 서브 앱으로 마운트
            self.app.mount(f"/dataset/{name}", dataset_app)
            print(f"Mounted dataset '{name}' at /dataset/{name}")


def find_available_port(start_port: int, max_attempts: int = 10, host="localhost"):
    """사용 가능한 포트 찾기"""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("No available ports found in the given range")


@click.command()
@click.option("--host", default="localhost", help="Host address")
@click.option("--port", default=5055, help="Port number")
@click.option(
    "--auto-port/--no-auto-port", default=True, help="Auto find available port"
)
@click.option("--duckdb", default="wasm", help="DuckDB connection mode")
@click.option("--static", default=None, help="Custom static files path")
def main(host, port, auto_port, duckdb, static):
    """Multi-Dataset Embedding Atlas CLI"""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 정적 파일 경로 설정
    if static is None:
        backend_path = (
            Path(__file__).parent
            / "packages"
            / "backend"
            / "embedding_atlas"
            / "static"
        )
        static = str(backend_path.resolve())

    print(f"Using static files from: {static}")

    # Multi-Atlas 서버 초기화
    atlas = MultiDatasetAtlas(static_path=static, duckdb_uri=duckdb)

    parquet_data_dir = Path(__file__).parent / "parquet_data"

    # Parquet 파일들 자동 감지 및 추가
    # 엔드포인트 이름, 파일명, 벡터 컬럼명
    parquet_files = [
        ("scouting_report_bgem3", "scouting_report_bgem3_with_year.parquet", "vector"),
        (
            "scouting_report_openai",
            "scouting_report_openai_with_year.parquet",
            "vector",
        ),
        (
            "scouting_report_qwen8b",
            "scouting_report_qwen8b_with_year.parquet",
            "vector",
        ),
    ]

    for name, file_name, vector_col in parquet_files:
        file_path = parquet_data_dir / file_name
        if file_path.exists():
            try:
                atlas.add_dataset(name, str(file_path), vector_col)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    if not atlas.datasets:
        print("No datasets loaded! Please check your parquet files.")
        return

    # 데이터셋별 라우트 마운트
    atlas.mount_dataset_routes()

    # 포트 설정
    if auto_port:
        final_port = find_available_port(port, max_attempts=10, host=host)
        if final_port != port:
            logging.info(f"Port {port} not available, using {final_port}")
    else:
        final_port = port

    print(f"\n🗺️  Multi-Dataset Embedding Atlas")
    print(f"📊 Loaded {len(atlas.datasets)} datasets:")
    for name in atlas.datasets.keys():
        print(f"   • {name}: http://{host}:{final_port}/dataset/{name}")
    print(f"📋 All datasets: http://{host}:{final_port}/datasets")
    print(f"🌐 Server starting at http://{host}:{final_port}")

    # 서버 실행
    uvicorn.run(atlas.app, host=host, port=final_port, access_log=False)


if __name__ == "__main__":
    main()
