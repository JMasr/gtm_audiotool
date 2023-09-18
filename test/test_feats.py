import os
from unittest import TestCase

import torch

from src.feats import FeatureExtractor
from src.utils import load_config_from_json


class TestFeatureExtractor(TestCase):

    def setUp(self):
        # Load configuration
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.feats_config = load_config_from_json(os.path.join(root_path, 'config', 'config_feat.json'))
        self.wav_path = os.path.join(root_path, 'data', 'test.wav')

    def test__read_audio(self):
        # Set configuration
        self.feats_config['feature_type'] = "MFCC"
        self.feats_config['extra_features'] = True

        # Prepare feature extractor
        feat_extractor = FeatureExtractor(self.feats_config)
        data, sample_rate = feat_extractor._read_audio(self.wav_path)

        self.assertIsInstance(sample_rate, int)
        self.assertIsInstance(data, torch.Tensor)

    def test_extract(self):
        # Check MFCC
        if not self.feats_config.get('feature_type') == 'MFCC':
            self.feats_config['feature_type'] = "MFCC"
            self.feats_config['extra_features'] = True

            feat_extractor = FeatureExtractor(self.feats_config)
            data = feat_extractor.extract(self.wav_path)

            self.assertIsInstance(data, torch.Tensor)
            self.assertEqual(data.shape[1], 121)

        # Check MelSpec
        if not self.feats_config.get('feature_type') == 'MelSpec':
            self.feats_config['feature_type'] = "MelSpec"
            self.feats_config['extra_features'] = True

            feat_extractor = FeatureExtractor(self.feats_config)
            data = feat_extractor.extract(self.wav_path)

            self.assertIsInstance(data, torch.Tensor)
            self.assertEqual(data.shape[1], 217)

        # Check logMelSpec
        if not self.feats_config.get('feature_type') == 'logMelSpec':
            self.feats_config['feature_type'] = "logMelSpec"
            self.feats_config['extra_features'] = True

            feat_extractor = FeatureExtractor(self.feats_config)
            data = feat_extractor.extract(self.wav_path)

            self.assertIsInstance(data, torch.Tensor)
            self.assertEqual(data.shape[1], 217)

        # Check ComParE_2016_voicing
        if not self.feats_config.get('feature_type') == 'ComParE_2016_energy':
            self.feats_config['feature_type'] = 'ComParE_2016_energy'
            self.feats_config['extra_features'] = False

            feat_extractor = FeatureExtractor(self.feats_config)
            data = feat_extractor.extract(self.wav_path)

            self.assertIsInstance(data, torch.Tensor)
            self.assertEqual(data.shape[1], 12)
