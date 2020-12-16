import numpy as np
import chess

class State():
    def __init__(self, self_play=False, color=chess.WHITE, board=None):
        # color one of chess.WHITE, chess.BLACK
        # default white
        self.color = color

        self.self_play = self_play

        if board is not None:
            self.board = board
        else:
            self.board = chess.Board()

    def serialize(self, pgn):
        pass

    def search_moves(self):
        moves = list(self.board.legal_moves)
        move_tuples = []
        for i in moves:
            # search moves, plug into model
            # represent as tuples with value and move
            pass
        return move_tuples[0]

    def move(self, move):
        if move in list(self.board.legal_moves):
            self.board.push(move)

    def play(self):
        if self.self_play == True:
            # self play logic
            pass
        else:
            # turn based logic
            pass








if __name__ == "__main__":
    s = State()

