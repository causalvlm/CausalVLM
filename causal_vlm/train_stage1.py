"""Run the three-phase captioning curriculum. Launch from the repository root."""
import argparse
from .src.training import main


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='causal_vlm/configs/stage1.yaml')
    p.add_argument('--data')
    p.add_argument('--encoder-checkpoint', help='Individual Perception encoder path or model ID')
    p.add_argument('--llm-checkpoint', help='Individual Llama path or model ID')
    p.add_argument('--encoder-loader', choices=['transformers', 'timm'])
    p.add_argument('--save-dir')
    p.add_argument('--init-checkpoint', help='Load weights only; starts a new curriculum')
    return p


if __name__ == '__main__':
    main(parser().parse_args(), 1)
