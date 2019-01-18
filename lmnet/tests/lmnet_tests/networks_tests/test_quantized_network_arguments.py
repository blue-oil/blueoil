from lmnet.networks.classification.lmnet_quantize import LmnetQuantize


def test_required_arguments():
    quantizer = LmnetQuantize()

    assert quantizer.first_layer_name is not None
    assert quantizer.last_layer_name is not None


if __name__ == '__main__':
    test_required_arguments()
