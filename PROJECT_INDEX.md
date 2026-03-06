# Embedding Atlas - Project Index

*Comprehensive documentation and navigation guide for the Embedding Atlas monorepo*

---

## 📖 Quick Navigation

- [🏗️ Architecture Overview](#-architecture-overview)
- [📦 Package Structure](#-package-structure)
- [🔧 Development Workflows](#-development-workflows)
- [🚀 Usage Patterns](#-usage-patterns)
- [🔗 Cross-References](#-cross-references)

---

## 🏗️ Architecture Overview

Embedding Atlas is an interactive visualization tool for large embeddings, built as a **monorepo** with npm workspaces containing both **frontend** (JavaScript/TypeScript) and **backend** (Python) components.

### Core Philosophy
- **Interactive Visualization**: Real-time exploration of high-dimensional embedding data
- **Multi-Modal Rendering**: WebGPU with WebGL2 fallback for optimal performance
- **Cross-Platform**: Web components, Python backend, and CLI tools
- **Scalable**: Handles datasets up to millions of points

### Technology Stack
```
Frontend: Svelte 5 + TypeScript + WebGPU/WebGL2 + D3
Backend:  Python + FastAPI + Pandas + UMAP
Data:     Parquet + DuckDB (WASM & Server)
Build:    Vite + npm workspaces + Rust/WASM
```

---

## 📦 Package Structure

### 🎨 Frontend Packages

#### `packages/component/` - Core Visualization Components
- **Purpose**: Core `EmbeddingView` and `EmbeddingViewMosaic` Svelte components
- **Key Files**:
  - `src/lib/embedding_view/` - Main visualization logic
  - `src/lib/webgpu_renderer/` - WebGPU rendering pipeline
  - `src/lib/webgl2_renderer/` - WebGL2 fallback renderer
- **Dependencies**: Svelte 5, D3, Mosaic Core
- **Exports**: TypeScript types, Svelte components

#### `packages/table/` - Data Table Component
- **Purpose**: Filterable, sortable data table with virtual scrolling
- **Key Files**:
  - `src/lib/Table.svelte` - Main table component
  - `src/lib/controllers/` - Scroll and resize controllers
  - `src/lib/mosaic-clients/` - DuckDB integration
- **Dependencies**: Svelte 5, Mosaic SQL
- **Exports**: Table component, custom cell/header APIs

#### `packages/viewer/` - Main Application
- **Purpose**: Complete frontend application and `EmbeddingAtlas` component
- **Key Files**:
  - Main frontend application
  - Static file build output (`dist/`)
- **Dependencies**: Component, Table packages
- **Exports**: Built static files for Python backend

#### `packages/embedding-atlas/` - Published NPM Package
- **Purpose**: Aggregates all frontend components for external use
- **Key Files**:
  - `src/react.ts` - React wrapper exports
  - `src/viewer.ts` - Main viewer exports
- **Dependencies**: All frontend workspace packages
- **Exports**: Unified API for React/Svelte/vanilla JS

### ⚙️ Low-Level Packages

#### `packages/umap-wasm/` - UMAP WebAssembly
- **Purpose**: High-performance UMAP implementation using umappp C++ library
- **Technology**: WebAssembly (C++)
- **Exports**: WASM UMAP functions
- **Testing**: ✅ Has test suite

#### `packages/density-clustering/` - Clustering Algorithm
- **Purpose**: Density clustering algorithm for automatic labeling
- **Technology**: Rust compiled to WebAssembly
- **Exports**: Clustering functions
- **Testing**: ✅ Has test suite

### 🐍 Backend Package

#### `packages/backend/` - Python Server & CLI
- **Purpose**: FastAPI server, CLI tools, and Jupyter widget
- **Key Modules**:
  - `embedding_atlas/cli.py` - Original CLI interface
  - `embedding_atlas/server.py` - FastAPI server
  - `embedding_atlas/data_source.py` - Data management layer
  - `embedding_atlas/projection.py` - UMAP/t-SNE computation
  - `embedding_atlas/widget.py` - Jupyter widget integration
- **Dependencies**: FastAPI, Pandas, UMAP-learn, DuckDB
- **Exports**: `embedding-atlas` CLI command, Python API

### 🔧 Custom Extensions

#### Root-Level CLI Scripts
- **`multi_atlas_cli.py`** - Serves multiple datasets simultaneously
- **`qdrant_dataload.py`** - Qdrant vector database integration

### 📚 Documentation & Examples

#### `packages/docs/` - Documentation Website
- **Purpose**: Static documentation site
- **Technology**: Documentation framework
- **Outputs**: <https://apple.github.io/embedding-atlas>

#### `packages/examples/` - Usage Examples
- **Purpose**: Example implementations and demos
- **Key Files**:
  - `src/react/` - React integration examples
  - `src/svelte/` - Svelte component examples
- **Dependencies**: All frontend packages

---

## 🔧 Development Workflows

### 🏗️ Build System
```bash
# Build everything (frontend + backend)
npm run build                    # Builds all npm packages
./scripts/build.sh              # Comprehensive build script

# Build individual packages
cd packages/component && npm run package
cd packages/viewer && npm run build
cd packages/backend && ./build.sh
```

### 🧪 Testing
```bash
npm run test                     # Run all tests
./scripts/test.sh               # Comprehensive test script

# Package-specific tests (only density-clustering and umap-wasm have tests)
cd packages/density-clustering && npm run test
cd packages/umap-wasm && npm run test
```

### 🚀 Development Servers
```bash
# Frontend development
cd packages/viewer && npm run dev

# Original Python CLI
cd packages/backend && python -m embedding_atlas.cli <dataset.parquet>

# Custom multi-dataset server
python multi_atlas_cli.py
```

---

## 🚀 Usage Patterns

### 🐍 Python Usage
```python
# CLI Tool
pip install embedding-atlas
embedding-atlas dataset.parquet

# Jupyter Widget
from embedding_atlas.widget import EmbeddingAtlasWidget
EmbeddingAtlasWidget(df)
```

### 📦 NPM Package Usage
```javascript
// Vanilla JavaScript
import { EmbeddingAtlas, EmbeddingView, Table } from "embedding-atlas";

// React
import { EmbeddingAtlas, EmbeddingView, Table } from "embedding-atlas/react";

// Svelte
import { EmbeddingAtlas, EmbeddingView, Table } from "embedding-atlas/svelte";
```

### 🗃️ Data Requirements
- **Primary Format**: Parquet files with vector embeddings
- **Required Columns**: Vector embeddings (as lists/arrays)
- **Optional Columns**:
  - `text` (for hover tooltips)
  - `id` (unique identifiers)
  - `x`, `y` (existing projections)
- **Auto-Detection**: Text columns (text, content, description, etc.)

---

## 🔗 Cross-References

### 📊 Data Flow Architecture
```
Parquet Files → DataSource → Projection (UMAP) → FastAPI → Frontend → WebGPU/WebGL2
     ↓              ↓            ↓            ↓         ↓           ↓
  pandas      caching layer   umappp      REST API   Svelte    GPU rendering
```

### 🏗️ Package Dependencies
```
embedding-atlas (published) ← viewer ← component
                           ← table
                           ← umap-wasm
                           ← density-clustering

backend (Python) → static files from viewer/dist/
```

### 🔧 Build Dependencies
- **Frontend**: npm workspaces, Vite, TypeScript, Svelte 5
- **Backend**: Python packaging, FastAPI, static file copying
- **WASM**: Rust (density-clustering), C++ (umap-wasm)
- **Integration**: Static files copied from `packages/viewer/dist/` to `packages/backend/embedding_atlas/static/`

### 🌐 Server Architecture
- **FastAPI**: Serves both data APIs and static frontend files
- **DuckDB**: Efficient data queries (browser WASM or server-side)
- **CORS**: Enabled for cross-origin requests
- **Auto-Port**: Automatic port detection to avoid conflicts

### 🎯 Key Technical Features
- **Automatic Clustering & Labeling**: Real-time data structure visualization
- **Kernel Density Estimation**: Density contours and outlier detection
- **Order-Independent Transparency**: Accurate overlapping point rendering
- **Real-Time Search**: Nearest neighbor and similarity search
- **Multi-Coordinated Views**: Interactive linking across metadata columns
- **Performance**: Up to millions of points with modern rendering

---

## 📋 Development Checklist

### ✅ Code Quality Standards
- [ ] TypeScript strict mode enabled
- [ ] Svelte 5 best practices followed
- [ ] Python type hints where applicable
- [ ] WebGPU with WebGL2 fallback implemented
- [ ] Error handling for WASM loading
- [ ] Performance monitoring for large datasets

### 🧪 Testing Coverage
- [x] density-clustering (Rust) - Has tests
- [x] umap-wasm (C++/WASM) - Has tests
- [ ] component (Svelte) - Needs test coverage
- [ ] table (Svelte) - Needs test coverage
- [ ] backend (Python) - Needs test coverage

### 📚 Documentation Status
- [x] README.md - Complete
- [x] CLAUDE.md - Complete development guide
- [x] PROJECT_INDEX.md - This comprehensive index
- [ ] API documentation - Needs generation
- [ ] Component documentation - Needs expansion

---

*Generated by Claude Code SuperClaude Framework | Last Updated: 2025-09-17*