build_dir="$( cd "$( dirname "$0" )"/.. && pwd )"
cd $build_dir
make -j4 derive_threshold
./derive_threshold.elf include/thresholds.h hls/include/hls_thresholds.h
echo 'Automatically exported include/thresholds.h and hls/include/hls_thresholds.h'
