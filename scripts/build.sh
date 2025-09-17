#!/bin/bash

# Build all JavaScript and Python packages

set -euxo pipefail

pushd packages/component
npm run build
popd

pushd packages/table
npm run build
popd

pushd packages/viewer
npm run build
popd

pushd packages/embedding-atlas
npm run build
popd

pushd packages/examples
npm run build
popd

set +e
pushd packages/backend
./build.sh || true
popd
set -e

pushd packages/docs
npm run build
popd
