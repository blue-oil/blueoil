EXP_ID=$1
DEVICE_ID=${2:-6}
rm -rf tmp/debug_c_$EXP_ID.prj
python tmp/convert/build.py --device_id 6 saved/debug_c_$EXP_ID/checkpoints/save.ckpt-1
