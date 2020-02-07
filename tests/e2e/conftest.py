import os
from pathlib import Path
import tempfile

import pytest

from blueoil.cmd.convert import convert
from blueoil.cmd.predict import predict
from blueoil.cmd.train import train
from lmnet import environment
from lmnet.utils.config import load


@pytest.fixture
def init_env():
    """Initialize blueoil environment"""
    blueoil_dir = str(Path('{}/../../../'.format(__file__)).resolve())
    config_dir = os.path.join(blueoil_dir, 'tests/fixtures/configs')

    train_output_dir = tempfile.TemporaryDirectory()
    predict_output_dir = tempfile.TemporaryDirectory()

    environment_originals = {}
    environ_originals = {}

    # TODO: Remove this setting after lmnet.environment has been refactored.
    envs = {
        "DATA_DIR": os.path.join(blueoil_dir, "lmnet", "tests", "fixtures", "datasets"),
        "OUTPUT_DIR": train_output_dir.name,
        "_EXPERIMENT_DIR": os.path.join(train_output_dir.name, "{experiment_id}"),
        "_TENSORBOARD_DIR": os.path.join(train_output_dir.name, "{experiment_id}", "tensorboard"),
        "_CHECKPOINTS_DIR": os.path.join(train_output_dir.name, "{experiment_id}", "checkpoints"),
    }

    for k, v in envs.items():
        environment_originals[k] = getattr(environment, k)
        environ_originals[k] = os.environ.get(k)
        setattr(environment, k, v)
        os.environ[k] = v

    yield {
        "train_output_dir": train_output_dir.name,
        "predict_output_dir": predict_output_dir.name,
        "blueoil_dir": blueoil_dir,
        "config_dir": config_dir,
    }

    for k, v in environment_originals.items():
        setattr(environment, k, v)
    for k, v in environ_originals.items():
        if v is not None:
            os.environ[k] = v
        else:
            del os.environ[k]

    train_output_dir.cleanup()
    predict_output_dir.cleanup()


def run_all_steps(dirs, config_file):
    """
    Test of the following steps.

    - Train using given config.
    - Convert using training result.
    - Predict using training result.
    """
    config_path = os.path.join(dirs["config_dir"], config_file)
    config = load(config_path)

    # Train
    # TODO: Remove this setting after lmnet.environment has been refactored.
    environment._init_flag = False
    experiment_id, checkpoint_name = train(config_path)

    train_output_dir = os.path.join(dirs["train_output_dir"], experiment_id)
    assert os.path.exists(os.path.join(train_output_dir, 'checkpoints'))

    # Convert
    # TODO: Remove this setting after lmnet.environment has been refactored.
    environment._init_flag = False
    convert(experiment_id)

    convert_output_dir = os.path.join(train_output_dir, 'export', checkpoint_name)
    lib_dir = os.path.join(
        convert_output_dir,
        "{}x{}".format(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        'output',
        'models',
        'lib',
    )
    assert os.path.exists(os.path.join(lib_dir, 'lib_aarch64.so'))
    assert os.path.exists(os.path.join(lib_dir, 'lib_arm.so'))
    assert os.path.exists(os.path.join(lib_dir, 'lib_fpga.so'))
    assert os.path.exists(os.path.join(lib_dir, 'lib_x86.so'))
    assert os.path.exists(os.path.join(lib_dir, 'lm_aarch64.elf'))
    assert os.path.exists(os.path.join(lib_dir, 'lm_arm.elf'))
    assert os.path.exists(os.path.join(lib_dir, 'lm_fpga.elf'))
    assert os.path.exists(os.path.join(lib_dir, 'lm_x86.elf'))

    # Predict
    predict_input_dir = os.path.join(dirs["blueoil_dir"], "lmnet/tests/fixtures/sample_images")
    predict_output_dir = dirs["predict_output_dir"]

    # TODO: Remove this setting after lmnet.environment has been refactored.
    environment._init_flag = False
    predict(predict_input_dir, predict_output_dir, experiment_id, checkpoint_name)

    assert os.path.exists(os.path.join(predict_output_dir, 'images'))
    assert os.path.exists(os.path.join(predict_output_dir, 'json', '0.json'))
    assert os.path.exists(os.path.join(predict_output_dir, 'npy', '0.npy'))
