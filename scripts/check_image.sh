#!/usr/bin/env bash
# Prompt 9 hardening: build the project's Docker image and verify its
# final (compressed, as it would actually be pushed/pulled) size stays
# under the project's 10GB hard constraint.
#
# Usage:
#   scripts/check_image.sh [image-tag]
#
# Exit codes:
#   0 - build succeeded and compressed image size is under the limit
#   1 - build failed, docker unavailable, or image exceeds the size limit
#
# This intentionally checks the COMPRESSED size (`docker save | gzip`),
# not `docker images`'s reported size, since that's what actually matters
# for a "10GB image" constraint in most submission/transfer contexts —
# the on-disk uncompressed size can look fine while the layered image
# still ships an oversized tarball once dependencies like torch/onnxruntime
# are in the mix.

set -euo pipefail

IMAGE_TAG="${1:-video-captioning-agent:check}"
MAX_BYTES=$((10 * 1000 * 1000 * 1000))  # 10GB, decimal (matches how cloud
                                          # registries typically report image
                                          # size) — intentionally NOT 10 *
                                          # 1024^3, to fail closed rather than
                                          # pass a borderline image due to a
                                          # binary/decimal mismatch.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "[check_image] FATAL: docker is not installed/on PATH" >&2
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "[check_image] FATAL: no Dockerfile found in $REPO_ROOT" >&2
    exit 1
fi

echo "[check_image] building ${IMAGE_TAG} ..."
if ! docker build -t "${IMAGE_TAG}" .; then
    echo "[check_image] FATAL: docker build failed" >&2
    exit 1
fi

TMP_TAR="$(mktemp /tmp/check_image.XXXXXX.tar.gz)"
cleanup() {
    rm -f "${TMP_TAR}"
}
trap cleanup EXIT

echo "[check_image] saving + compressing image to measure real transfer size ..."
if ! docker save "${IMAGE_TAG}" | gzip > "${TMP_TAR}"; then
    echo "[check_image] FATAL: docker save/gzip failed" >&2
    exit 1
fi

COMPRESSED_BYTES=$(wc -c < "${TMP_TAR}")
UNCOMPRESSED_BYTES=$(docker image inspect "${IMAGE_TAG}" --format '{{.Size}}' 2>/dev/null || echo "unknown")

human_readable() {
    # Portable-ish bytes -> human string without depending on `numfmt`
    # being present in every environment this script might run in.
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN {
        split("B KB MB GB TB", units, " ");
        i = 1;
        while (b >= 1000 && i < 5) { b /= 1000; i++ }
        printf "%.2f %s", b, units[i]
    }'
}

echo "[check_image] image:               ${IMAGE_TAG}"
echo "[check_image] uncompressed size:    ${UNCOMPRESSED_BYTES} bytes ($(human_readable "${UNCOMPRESSED_BYTES}" 2>/dev/null || echo unknown))"
echo "[check_image] compressed (tar.gz):  ${COMPRESSED_BYTES} bytes ($(human_readable "${COMPRESSED_BYTES}"))"
echo "[check_image] limit:                ${MAX_BYTES} bytes ($(human_readable "${MAX_BYTES}"))"

if [ "${COMPRESSED_BYTES}" -ge "${MAX_BYTES}" ]; then
    echo "[check_image] FAIL: compressed image size ${COMPRESSED_BYTES} bytes exceeds the 10GB limit" >&2
    exit 1
fi

echo "[check_image] PASS: compressed image size is under the 10GB limit"
exit 0
