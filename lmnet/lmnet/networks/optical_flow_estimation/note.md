# Note

## run docker images
```
docker run -it -v /storage/ki42/blueoil:/home/blueoil -v /nfs01/dataset/optical_flow_estimation:/home/blueoil/dataset blueoil_ki42:v0.10.0-42-g86efe70
```

## test commands

### dataset_loader
```
cd lmnet
pytest tests/lmnet_tests/datasets_tests/test_optical_flow_estimation.py
```

### network_model
```
cd lmnet
CUDA_VISIBLE_DEVICES=XXX pytest tests/lmnet_tests/networks_tests/optical_flow_estimation_tests/test_flownet_s_v1.py

```

## training commands

```
CUDA_VISIBLE_DEVICES=XXX python lmnet/executor/train.py -c lmnet/configs/core/optical_flow_estimation/lm_flownet_v1.py -i YYY
```


## pb conversion

```
CUDA_VISIBLE_DEVICES=XXX python lmnet/executor/export.py -i YYY --restore_path ZZZ
```

## dlk conversion

```
cd dlk
PYTHONPATH=python/dlk python python/dlk/scripts/generate_project.py -i minimal_graph_with_shape.pb -o ../tmp/ -p XXX -hq -ts
```

## makefile
```
export OMP_NUM_THREADS=20
make lib_x86_avx -j8
```

```
mkdir build
cd build
cmake .. -DUSE_AVX=1
make -j8 lm
```
