import numpy as np
import random
import chess

class State():
    def __init__(self, self_play=False, color=chess.WHITE, board=None):
        # color one of chess.WHITE, chess.BLACK
        # default white
        self.color = color

        self.states = [] # change this if parsing from pgn

        self.self_play = self_play

        if board is not None:
            self.board = board
        else:
            self.board = chess.Board()

    def fen_to_bits(self, pgn=None):
        converted_board = [] # np.zeros(shape of data)
        if pgn == None:
            # use board.pieces to get int representations of the board
            # alternating white->black, P R N B Q K
            for i in range(1, 7):
                converted_board.append(int(self.board.pieces(i, chess.WHITE)))
                converted_board.append(int(self.board.pieces(i, chess.BLACK)))
        else:
            # used for parsing games from pgn, might not be needed
            pass
        return converted_board

    def search_moves(self):
        moves = list(self.board.legal_moves)
        move_tuples = []
        # replace random move generation with evaluator
        move_to_play = random.randint(0, len(moves) - 1)
        for i in range(0, len(moves)):
            self.board.push(moves[i])
            # search moves, plug into model
            # use and evaluator class
            # represent as tuples with value and move
            self.board.pop()
            pass
        return moves[move_to_play]# move_tuples[0]

    def move(self, move):
        self.states.append(self.fen_to_bits())
        self.board.push(move)

    def play(self):
        while not self.board.is_game_over():
            print('\n')
            print({True: 'White', False: "Black"}[self.board.turn] + " to play")
            print(self.board.unicode())
            # save the board at each position
            print('\n')
            if self.self_play == True:
                move_to_play = self.search_moves()
                self.move(move_to_play)
            else:
                if self.board.turn == self.color:
                    move_to_play = self.search_moves()
                    self.move(move_to_play)
                else:
                    # change this input to a form or something
                    human_move = input("enter a move: ")
                    try:
                        human_move = self.board.parse_san(human_move)
                    except ValueError:
                        print('invalid move')
                        continue
                    self.move(human_move)
        result = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}[self.board.result()]
        for i in range(0, len(self.states)):
            self.states[i] = (result, self.states[i])
        print(self.states)
        return self.states
