# Convert your training result to FPGA ready format

```
$ ./blueoil.sh convert config/test.yml ./saved/test_20180101000000

Usage:
  ./blueoil.sh convert <YML_CONFIG_FILE> <EXPERIMENT_DIRECTORY> <CHECKPOINT_NO(optional)>

  Arguments:
    YML_CONFIG_FILE       config file path for this training  [required]
    EXPERIMENT_DIRECTORY  experiment directory path for input [required]
                          this is same as {OUTPUT_DIRECTORY}/{EXPERIMENT_ID} in training options.
    CHECKPOINT_NO         checkpoint number [optional] (default is latest checkpoint)
                          if you want to use save.ckpt-1000, you can set CHECKPOINT_NO as 1000.
```

`blueoil convert` command converts trained models to executable binary files for x86, ARM Cortex-A9, and FPGA.


