"""Run ten head-only epochs followed by ten joint-training epochs."""
from .train_stage1 import parser
from .src.training import main


if __name__ == '__main__':
    p = parser()
    p.description = __doc__
    p.set_defaults(config='causal_vlm/configs/stage2.yaml')
    p.add_argument('--stage1-ckpt', required=True)
    main(p.parse_args(), 2)
