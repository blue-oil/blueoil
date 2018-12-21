FROM masuda_blueoil:local_build

MAINTAINER masuda@leapmind.io

WORKDIR /home/blueoil

# Set env to install horovod with nccl and tensorflow option
ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_WITH_TENSORFLOW 1
# Set temporarily CUDA stubs to install Horovod
RUN ldconfig /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs
# Install requirements for distributed training
RUN pip install -r lmet/dist.requirements.txt
# Unset temporarily CUDA stubs
RUN ldconfig

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0 --mca btl_vader_single_copy_mechanism none
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
