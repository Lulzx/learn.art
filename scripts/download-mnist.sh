#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/mnist}"
base="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p "$destination"

files=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)
checksums=(
  f68b3c2dcbeaaa9fbdd348bbdeb94873
  d53e105ee54ea40749a09fcbcd1e9432
  9fb629c4189551a2d022fa330f9573f3
  ec29112dd5afa0611ce80d1b7f02629c
)

for index in "${!files[@]}"; do
  name="${files[$index]}"
  archive="$destination/$name.gz"
  raw="$destination/$name"
  if [[ ! -f "$archive" ]]; then
    curl --fail --location --retry 3 "$base/$name.gz" --output "$archive"
  fi
  actual="$(md5 -q "$archive")"
  if [[ "$actual" != "${checksums[$index]}" ]]; then
    echo "checksum mismatch for $archive" >&2
    exit 1
  fi
  if [[ ! -f "$raw" ]]; then
    gzip --decompress --stdout "$archive" > "$raw"
  fi
done

echo "MNIST IDX files are ready in $destination"
