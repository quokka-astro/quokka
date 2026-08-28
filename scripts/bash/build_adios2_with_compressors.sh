#!/usr/bin/env bash
# Build ADIOS2 with ZFP, SZ2, and BLOSC2 support from source.
set -euo pipefail

PREFIX=${1:-${ADIOS2_PREFIX:-"$PWD/local-adios2"}}
BUILD_ROOT=${BUILD_ROOT:-"$PWD/build-adios2"}
ADIOS2_TAG=${ADIOS2_TAG:-v2.11.0}
ZFP_TAG=${ZFP_TAG:-1.0.1}
SZ_TAG=${SZ_TAG:-v2.1.12.5}
BLOSC2_TAG=${BLOSC2_TAG:-v2.22.0}
ADIOS2_USE_MPI=${ADIOS2_USE_MPI:-ON}

detect_jobs() {
	if command -v sysctl >/dev/null && sysctl -n hw.ncpu >/dev/null 2>&1; then
		sysctl -n hw.ncpu
	elif command -v nproc >/dev/null; then
		nproc
	else
		echo 4
	fi
}
JOBS=${JOBS:-$(detect_jobs)}

GENERATOR=${GENERATOR:-}
if [[ -z "${GENERATOR}" ]] && command -v ninja >/dev/null; then
	GENERATOR="-G Ninja"
fi

echo "Install prefix : $PREFIX"
echo "Build root     : $BUILD_ROOT"
echo "CMake generator: ${GENERATOR:-Default}"
echo "Jobs           : $JOBS"

command -v cmake >/dev/null || { echo "cmake is required"; exit 1; }
DOWNLOAD_TOOL=""
if command -v curl >/dev/null; then
	DOWNLOAD_TOOL="curl -fL"
elif command -v wget >/dev/null; then
	DOWNLOAD_TOOL="wget -O -"
else
	echo "curl or wget is required to download tarballs."
	exit 1
fi

mkdir -p "$BUILD_ROOT" "$PREFIX"

download_file() {
	local url=$1 dest=$2
	if [[ -s "$dest" ]]; then
		return 0
	fi
	echo "Downloading $url" >&2
	if [[ "$DOWNLOAD_TOOL" == curl* ]]; then
		$DOWNLOAD_TOOL "$url" -o "$dest"
	else
		$DOWNLOAD_TOOL "$url" >"$dest"
	fi
}

download_optional() {
	local url=$1 dest=$2
	set +e
	if [[ "$DOWNLOAD_TOOL" == curl* ]]; then
		$DOWNLOAD_TOOL "$url" -o "$dest"
	else
		$DOWNLOAD_TOOL "$url" >"$dest"
	fi
	local status=$?
	set -e
	return $status
}

fetch_tarball() {
	local repo=$1 tag=$2 name=$3
	local tarball="$BUILD_ROOT/${name}-${tag}.tar.gz"
	local tar_url="https://github.com/${repo}/archive/refs/tags/${tag}.tar.gz"
	download_file "$tar_url" "$tarball"

	local attestation=""
	for candidate in \
		"https://github.com/${repo}/releases/download/${tag}/attestation.intoto.jsonl" \
		"https://github.com/${repo}/releases/download/${tag}/${name}-${tag}-attestation.intoto.jsonl" \
		"https://github.com/${repo}/releases/download/${tag}/${tag}-attestation.intoto.jsonl"; do
		local out="$BUILD_ROOT/${name}-${tag}.intoto.jsonl"
		if download_optional "$candidate" "$out"; then
			attestation="$out"
			echo "Downloaded attestation $candidate" >&2
			break
		else
			rm -f "$out"
		fi
	done

	if [[ -n "$attestation" ]]; then
		if command -v slsa-verifier >/dev/null; then
			echo "Verifying attestation for ${name} ${tag}" >&2
			slsa-verifier verify-artifact \
				--provenance-path "$attestation" \
				--source-uri "github.com/${repo}" \
				--source-tag "$tag" \
				"$tarball" >&2
		else
			echo "Attestation present but slsa-verifier not installed; skipping verification." >&2
		fi
	fi

	echo "$tarball"
}

extract_tarball() {
	local tarball=$1 name=$2 tag=$3
	local stripped=${tag#v}
	local guessed="$BUILD_ROOT/${name}-${stripped}"
	if [[ ! -d "$guessed" ]]; then
		tar -xzf "$tarball" -C "$BUILD_ROOT"
	fi
	if [[ -d "$guessed" ]]; then
		echo "$guessed"
		return
	fi
	local top
	top=$(tar -tzf "$tarball" | head -1 | cut -d/ -f1)
	echo "$BUILD_ROOT/$top"
}

cmake_build_install() {
	local src=$1
	local bld=$2
	shift 2
	cmake -S "$src" -B "$bld" ${GENERATOR:+$GENERATOR} \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX="$PREFIX" \
		"$@"
	cmake --build "$bld" --target install --config Release -- -j "$JOBS"
}

echo ">>> Building ZFP $ZFP_TAG"
zfp_tar=$(fetch_tarball LLNL/zfp "$ZFP_TAG" zfp)
zfp_src=$(extract_tarball "$zfp_tar" zfp "$ZFP_TAG")
cmake_build_install "$zfp_src" "$BUILD_ROOT/zfp-build" \
	-DBUILD_SHARED_LIBS=ON \
	-DBUILD_TESTING=OFF \
	-DBUILD_EXAMPLES=OFF

echo ">>> Building SZ $SZ_TAG"
sz_tar=$(fetch_tarball szcompressor/SZ2 "$SZ_TAG" SZ2)
sz_src=$(extract_tarball "$sz_tar" SZ2 "$SZ_TAG")
cmake_build_install "$sz_src" "$BUILD_ROOT/sz-build" \
	-DBUILD_SHARED_LIBS=ON \
	-DBUILD_TESTS=OFF \
	-DBUILD_EXAMPLES=OFF \
	-DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo ">>> Building BLOSC2 $BLOSC2_TAG"
blosc_tar=$(fetch_tarball Blosc/c-blosc2 "$BLOSC2_TAG" c-blosc2)
blosc_src=$(extract_tarball "$blosc_tar" c-blosc2 "$BLOSC2_TAG")
cmake_build_install "$blosc_src" "$BUILD_ROOT/blosc2-build" \
	-DBUILD_SHARED=ON \
	-DBUILD_STATIC=OFF \
	-DBUILD_TESTS=OFF \
	-DBUILD_BENCHMARKS=OFF \
	-DBUILD_EXAMPLES=OFF \
	-DPREFER_EXTERNAL_ZLIB=ON

echo ">>> Building ADIOS2 $ADIOS2_TAG"
adios_tar=$(fetch_tarball ornladios/ADIOS2 "$ADIOS2_TAG" ADIOS2)
adios_src=$(extract_tarball "$adios_tar" ADIOS2 "$ADIOS2_TAG")
cmake_build_install "$adios_src" "$BUILD_ROOT/adios2-build" \
	-DCMAKE_PREFIX_PATH="$PREFIX" \
	-DADIOS2_BUILD_EXAMPLES=OFF \
	-DADIOS2_BUILD_TESTING=OFF \
	-DADIOS2_USE_MPI="$ADIOS2_USE_MPI" \
	-DADIOS2_USE_Blosc=ON \
	-DADIOS2_USE_ZFP=ON \
	-DADIOS2_USE_SZ=ON \
	-DADIOS2_USE_Python=OFF \
	-DADIOS2_USE_Fortran=OFF

echo "Done. Add $PREFIX/bin to your PATH and $PREFIX/lib to your LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on macOS)."
