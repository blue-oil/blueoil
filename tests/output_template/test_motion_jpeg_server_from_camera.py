from unittest import TestCase

from motion_jpeg_server_from_camera import _visualizer


class TestMotionJpegServer(TestCase):

    def test_visualizer_classification_support(self):
        self.assertTrue(_visualizer("IMAGE.CLASSIFICATION"))

    def test_visualizer_object_detection_support(self):
        self.assertTrue(_visualizer("IMAGE.OBJECT_DETECTION"))

    def test_visualizer_semantic_segmentation_support(self):
        self.assertTrue(_visualizer("IMAGE.SEMANTIC_SEGMENTATION"))

    def test_visualizer_not_supported(self):
        with self.assertRaises(ValueError):
            _visualizer("NOT_SUPPORTED")
