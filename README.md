# NEAT neural slime volleyball <a href="https://emoji.gg/emoji/4545_RainbowBlobCat"><img src="https://cdn3.emoji.gg/emojis/4545_RainbowBlobCat.gif" width="32px" height="32px" alt="RainbowBlobCat"></a>

Evolves an agent through tournament self-play using [NEAT-Python](https://github.com/google/evojax/blob/main/examples/train_slimevolley.py) on [SlimeVolley](https://github.com/google/evojax/blob/main/examples/train_slimevolley.py) task.

1. set NEAT parameters in `config-feedforward`
2. set training parameters in `train.py`
3. run `python3 train.py`

See output logs in `log/neat-slimevolley` for final model .gif, saved checkpoints, and network graphs.

Uses feature engineering, and hit reward annealing to speed up convergence.

## final agent + example evolved topology

![final_agent](https://github.com/ellenjxu/neat-slimevolley/assets/56745453/c2adadb9-70fc-4707-a521-ee4d553191ce)
![topology](https://github.com/ellenjxu/neuralslimevolley/assets/56745453/a4bfeded-e389-4c16-83ff-0caa453d83f4)
