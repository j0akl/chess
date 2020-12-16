# Chess Engine

Representation:
using self.board.pieces bitmask to represent pieces, will have to see if that
works for training or not. Dont think I need to represent checks or anything
because that will be handled by legal move generation. Over training it should
understand the whole game, need to test that though.

data is represented as an array of 12 boolean arrays length 64, these are the
bitboard representations of the state of the board. Each is paired with the result of
the game
