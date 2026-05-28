# AlphaZero And Go Research Notes

## Primary Sources

- AlphaGo Zero, DeepMind blog:
  https://deepmind.google/blog/alphago-zero-starting-from-scratch/
- AlphaGo Zero, Nature paper record:
  https://discovery.ucl.ac.uk/id/eprint/10045895/
- AlphaZero, DeepMind blog:
  https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/
- AlphaZero arXiv preprint:
  https://arxiv.org/abs/1712.01815
- OpenSpiel AlphaZero docs:
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- Minigo:
  https://github.com/tensorflow/minigo
- PyTorch Go/Gomoku AlphaZero reference:
  https://github.com/michaelnny/alpha_zero
- KataGo acceleration paper:
  https://arxiv.org/abs/1902.10565
- ELF OpenGo:
  https://arxiv.org/abs/1902.04522
- Tromp-Taylor concise Go rules:
  https://www.cs.cmu.edu/~wjh/go/tmp/rules/TrompTaylor.html
- AGA Go rules:
  https://www.britgo.org/rules/agarules.html
- Hugging Face custom PyTorch model upload docs:
  https://huggingface.co/docs/hub/models-uploading

## Implementation Conclusions

- The training loop should be self-play only.
- Use one policy-value network, not separate policy and value networks.
- Use MCTS during self-play to produce improved policy targets.
- Use direct legal-action argmax over policy logits for base inference.
- Keep pass as an explicit action because Go cannot terminate rigorously without it.
- Keep board logits as a first-class `[board_size, board_size]` view for the user's requested MOP.
- Start with an MLP over two board channels before adding ResNet variants.
- Track validation through match play, including trained direct-policy inference versus a randomly initialized model.
- Prefer area scoring for automated Go evaluation.
