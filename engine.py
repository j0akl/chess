import numpy as np
import random
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
        # replace random move generation with evaluator
        move_to_play = random.randint(0, len(moves) - 1)
        for i in moves:
            # search moves, plug into model
            # use and evaluator class
            # represent as tuples with value and move
            pass
        return moves[move_to_play]# move_tuples[0]

    def play(self):
        while not self.board.is_game_over():
            print('\n')
            print({True: 'White', False: "Black"}[self.board.turn])
            print(self.board.unicode())
            print('\n')
            if self.self_play == True:
                move_to_play = self.search_moves()
                self.board.push(move_to_play)
            else:
                if self.board.turn == self.color:
                    move_to_play = self.search_moves()
                    self.board.push(move_to_play)
                else:
                    # change this input to a form or something
                    human_move = input("enter a move: ")
                    try:
                        human_move = self.board.parse_san(human_move)
                    except ValueError:
                        print('invalid move')
                        continue
                    self.board.push(human_move)
        result = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}[self.board.result()]
        print(result)

if __name__ == "__main__":
    s = State(self_play=True)
    s.play()

