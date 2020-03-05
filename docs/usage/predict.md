# Making prediction
Make predictions from input dir images by using trained model.

Save the predictions npy, json, images results to output dir.
* npy: `{output_dir}/npy/{batch number}.npy`
* json: `{output_dir}/json/{batch number}.json`
* images: `{output_dir}/images/{some type}/{input image file name}`

The output predictions Tensor(npy) and json format depends on task type. Plsease see [Output Data Specification](../specification/output_data.md).

```
# python blueoil/cmd/main.py predict --help
Usage: main.py predict [OPTIONS]

  Predict by using trained model.

Options:
  -i, --input TEXT                Directory containing predicted images.
                                  [required]
  -o, --output TEXT               Directory to output prediction result.
                                  [required]
  -e, --experiment_id TEXT        ID of this experiment.  [required]
  -c, --config TEXT               Path of config file.
  -p, --checkpoint TEXT           Checkpoint name. e.g. save.ckpt-10001
  --save_images / --no_save_images
                                  Flag of saving images. Default is True.
  --help                          Show this message and exit.
```
