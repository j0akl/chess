# Chess Engine

Representation:
using self.board.pieces bitmask to represent pieces, will have to see if that
works for training or not. Dont think I need to represent checks or anything
because that will be handled by legal move generation. Over training it should
understand the whole game, need to test that though.

train model on value of each position based on the possible move

Start training from random moves
