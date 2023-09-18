import os
import argparse
from src.feats import FeatureExtractor
from src.utils import load_config_from_json

all_feats = ['MFCC', 'MelSpec', 'logMelSpec',
             'ComParE_2016_voicing', 'ComParE_2016_energy',
             'ComParE_2016_basic_spectral', 'ComParE_2016_spectral',
             'ComParE_2016_mfcc', 'ComParE_2016_rasta',
             'Spafe_mfcc', 'Spafe_imfcc', 'Spafe_cqcc', 'Spafe_gfcc', 'Spafe_lfcc',
             'Spafe_lpc', 'Spafe_lpcc', 'Spafe_msrcc', 'Spafe_ngcc', 'Spafe_pncc',
             'Spafe_psrcc', 'Spafe_plp', 'Spafe_rplp']


if __name__ == "__main__":
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", "-r", help="Path to root folder", default=None)
    parser.add_argument("--wav", "-w", help="Path to input audio file", default="data/test.wav")
    parser.add_argument("--output", "-o", help="Path to output file", default="results/")
    args = parser.parse_args()

    # Set paths
    root_path = args.r if args.r else os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(root_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Load configuration
    feats_config = load_config_from_json(os.path.join(root_path, 'config', 'config_feat.json'))

    # Set configuration
    feats_config['feature_type'] = "MFCC"
    feats_config['extra_features'] = True

    # Prepare feature extractor
    feat_extractor = FeatureExtractor(feats_config)

    # Extract features
    F = feat_extractor.extract(args.wav)
    print("Features extracted: ", F.shape)
