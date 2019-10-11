EXP_ID=$1
DEVICE_ID=${2:-6}
rm -rf saved/debug_c_$EXP_ID
CUDA_VISIBLE_DEVICES=$DEVICE_ID python lmnet/executor/train.py -c lmnet/configs/core/optical_flow_estimation/lm_flownet_debug_quantized.py -i debug_c_$EXP_ID
rm -rf tmp/debug_c_$EXP_ID.prj
python tmp/convert/build.py --device_id 6 saved/debug_c_$EXP_ID/checkpoints/save.ckpt-1
