#!/usr/bin/env python3
"""
Script to load data from Qdrant vector database and prepare it for Embedding Atlas visualization.
"""

import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter


def connect_to_qdrant(url: str = "http://localhost:6333") -> QdrantClient:
    """Connect to Qdrant client."""
    try:
        client = QdrantClient(url=url)
        # Test connection
        collections = client.get_collections()
        print(f"Successfully connected to Qdrant at {url}")
        print(f"Available collections: {[c.name for c in collections.collections]}")
        return client
    except Exception as e:
        print(f"Failed to connect to Qdrant at {url}: {e}")
        sys.exit(1)


def list_collections(client: QdrantClient) -> List[str]:
    """List all available collections."""
    collections = client.get_collections()
    return [c.name for c in collections.collections]


def extract_year_from_source(source: str) -> Optional[int]:
    """Extract year from source file path."""
    if not source or not isinstance(source, str):
        return None

    # 파일명에서 4자리 연도 찾기
    import re
    pattern = r'(?:19|20)\d{2}'  # 1900-2099 연도 패턴 (non-capturing group 사용)
    matches = re.findall(pattern, source)

    if matches:
        # 가장 마지막에 나오는 연도가 보통 맞음
        year = int(matches[-1])
        if 1900 <= year <= 2030:
            return year

    return None


def setup_text_field(df: pd.DataFrame) -> pd.DataFrame:
    """Setup text field for automatic clustering."""
    df_copy = df.copy()

    # 원본 텍스트 데이터를 text 필드로 사용 (자동 라벨링을 위해)
    original_text_fields = [
        "text",
        "content",
        "description",
        "document",
        "report",
        "summary",
        "scouting_report"
    ]

    for field in original_text_fields:
        if field in df_copy.columns and pd.notna(df_copy[field]).any():
            if field != "text":  # 이미 text 필드가 아닌 경우에만 복사
                df_copy["text"] = df_copy[field].fillna("No description")
                print(f"Using '{field}' as text field for clustering")
            else:
                print("Using existing 'text' field for clustering")
            break
    else:
        print("Warning: No text field found for clustering")

    return df_copy


def extract_collection_data(
    client: QdrantClient,
    collection_name: str,
    limit: Optional[int] = None,
    filter_condition: Optional[Filter] = None,
) -> pd.DataFrame:
    """Extract all data from a Qdrant collection."""
    print(f"Extracting data from collection: {collection_name}")

    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"Collection size: {collection_info.points_count}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")

        # Scroll through all points
        all_points = []
        offset = None
        batch_size = 1000
        total_extracted = 0

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
                scroll_filter=filter_condition,
            )

            if not points:
                break

            all_points.extend(points)
            total_extracted += len(points)
            print(f"Extracted {total_extracted} points...")

            if limit and total_extracted >= limit:
                all_points = all_points[:limit]
                break

            offset = next_offset
            if next_offset is None:
                break

        print(f"Total points extracted: {len(all_points)}")

        # Convert to DataFrame
        data = []
        for point in all_points:
            row = {
                "id": str(point.id),
                "vector": np.array(point.vector),
            }

            # Add payload data as metadata
            if point.payload:
                for key, value in point.payload.items():
                    # Handle different data types
                    if isinstance(value, (list, dict)):
                        row[key] = str(value)
                    else:
                        row[key] = value

            data.append(row)

        df = pd.DataFrame(data)

        # 연도 필드 매핑 (Qdrant 메타데이터에서 또는 source에서 추출)
        year_fields = ["scouting_year", "year", "season", "date_year"]
        found_year = False

        for year_field in year_fields:
            if year_field in df.columns:
                # year가 아닌 다른 필드를 year로 매핑하는 경우에만 변환
                if year_field != "year":
                    df["year"] = pd.to_numeric(df[year_field], errors="coerce")
                    print(f"Mapped '{year_field}' to year field")
                else:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")
                    print("Using existing 'year' field")
                found_year = True
                break

        if not found_year and "source" in df.columns:
            # source 필드에서 연도 추출
            print("Extracting year from source field...")
            df["year"] = df["source"].apply(extract_year_from_source)
            year_count = df["year"].notna().sum()
            print(f"Extracted year from source: {year_count}/{len(df)} records")
            found_year = year_count > 0

        # year 필드를 문자열로 변환 (categorical 처리를 위해)
        if "year" in df.columns and df["year"].notna().any():
            df["year"] = df["year"].astype(str).replace("nan", None)
            print("Converted year field to string for categorical filtering")


        if not found_year:
            print("No year information found")

        # Setup text field for clustering
        df = setup_text_field(df)

        # 연도별 간단한 통계
        if "year" in df.columns:
            year_count = df["year"].notna().sum()
            if year_count > 0:
                years = df["year"].dropna()
                year_range = f"{int(years.min())}-{int(years.max())}"
                print(f"Year data: {year_count}/{len(df)} records, range: {year_range}")

        print(f"DataFrame created with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # 핵심 필드 확인
        key_fields = ["text", "year", "vector"]
        existing_key = [field for field in key_fields if field in df.columns]
        if existing_key:
            print(f"Key fields: {existing_key}")

        return df

    except Exception as e:
        print(f"Error extracting data from collection {collection_name}: {e}")
        sys.exit(1)


def save_to_parquet(df: pd.DataFrame, output_path: str):
    """Save DataFrame to parquet file."""
    try:
        # Convert vector column to proper format for parquet
        if "vector" in df.columns:
            # Convert numpy arrays to lists for parquet compatibility
            df["vector"] = df["vector"].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

        df.to_parquet(output_path, index=False)
        print(f"Data saved to: {output_path}")

        # 파일 크기 확인을 os.path.getsize로 변경
        import os

        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"Error saving to parquet: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from Qdrant and prepare for Embedding Atlas with year integration"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection", help="Name of the Qdrant collection to extract"
    )
    parser.add_argument(
        "--output",
        default="qdrant_data.parquet",
        help="Output parquet file path (default: qdrant_data.parquet)",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of points to extract (default: all)"
    )
    parser.add_argument(
        "--year-filter",
        type=int,
        nargs="+",
        help="Filter by specific years (e.g., --year-filter 2020 2021 2022)",
    )
    parser.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Filter by year range (e.g., --year-range 2020 2023)",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List available collections and exit",
    )

    args = parser.parse_args()

    # Connect to Qdrant
    client = connect_to_qdrant(args.url)

    # List collections if requested
    if args.list_collections:
        collections = list_collections(client)
        print("\nAvailable collections:")
        for collection in collections:
            print(f"  - {collection}")
        return

    # Collection name is required if not listing collections
    if not args.collection:
        print("Error: --collection is required when not using --list-collections")
        parser.print_help()
        return

    # Extract data
    df = extract_collection_data(client, args.collection, args.limit)

    # Apply year filtering if requested
    if args.year_filter or args.year_range:
        original_size = len(df)

        if args.year_filter:
            df = df[df["year"].isin(args.year_filter)]
            print(
                f"Filtered to years {args.year_filter}: {len(df)} records (from {original_size})"
            )

        elif args.year_range:
            start_year, end_year = args.year_range
            df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
            print(
                f"Filtered to years {start_year}-{end_year}: {len(df)} records (from {original_size})"
            )

        if len(df) == 0:
            print("Warning: No data remaining after year filtering!")
            return

    # Save to parquet
    save_to_parquet(df, args.output)

    print(f"\n{'=' * 50}")
    print(f"✅ Data extraction complete!")
    print(f"{'=' * 50}")

    # 요약 정보 출력
    if "year" in df.columns:
        years = df["year"].dropna().unique()
        if len(years) > 0:
            year_range = f"{int(min(years))}-{int(max(years))}"
            print(f"📅 Data spans: {year_range} ({len(years)} different years)")

    print(f"📊 Total records: {len(df)}")
    print(f"💾 Saved to: {args.output}")

    print("\n🚀 To visualize with Embedding Atlas:")
    print("   embedding-atlas {} --vector vector".format(args.output))

    print("\n🔍 Key fields for filtering:")
    print("   • player_name: filter by specific player")
    print("   • year: filter by year")
    print("   • position: filter by playing position")
    print("   • team: filter by team")
    print("   • text: original scouting content for clustering")
    print("   • vector: embedding coordinates")

    # 텍스트 필드 길이 통계 표시 (원본 텍스트 품질 확인용)
    if "text" in df.columns:
        text_lengths = df["text"].astype(str).str.len()
        avg_length = text_lengths.mean()
        print(f"\n📝 Text data quality: avg {avg_length:.0f} chars per entry")


if __name__ == "__main__":
    main()
