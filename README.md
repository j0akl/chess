# Chess Engine

think of better representation for board state
 - needs to handle all rights (castling, en passant, etc)

Representation:

- turn: 1 bit
- positions of pieces
 - number each square, numbers 1-15 represent piece types white and black (?)
 - 1 byte for each square
- en passant: 16 bits
- castle: 4 bits
- check 2 bits
total: 534 (can be cut down?)

train model on value of each position based on the possible move

Start training from random moves, find a way to represent value of a position
attach value of game to each board state? might be the way to go
