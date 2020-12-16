import numpy as np
import random
import chess
import torch
from train import Net

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
                converted_board.append(self.board.pieces(i,
                                                         chess.WHITE).tolist())
                converted_board.append(self.board.pieces(i,
                                                         chess.BLACK).tolist())
        else:
            # used for parsing games from pgn, might not be needed
            pass
        return np.array(converted_board).astype(np.byte)

    def search_moves(self):
        moves = list(self.board.legal_moves)
        move_tuples = []
        for i in range(0, len(moves)):
            self.board.push(moves[i])
            move_tuples.append((self.eval_move().item(), moves[i]))
            self.board.pop()
        move_tuples.sort(reverse=True)
        if self.board.turn == chess.WHITE:
            return move_tuples[0][1]
        else:
            return move_tuples[-1][1]

    def eval_move(self):
        model = Net()
        model.load_state_dict(torch.load('model/v1.pt'))
        state = torch.tensor(self.fen_to_bits()).float().view(1, 12, 64) # shape
        prediction = model(state)
        return prediction



    def move(self, move):
        self.states.append(self.fen_to_bits())
        self.board.push(move)

    def play(self):
        while not self.board.is_game_over():
            # print('\n')
            # print({True: 'White', False: "Black"}[self.board.turn] + " to play")
            # print(self.board.unicode())
            # save the board at each position
            # print('\n')
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
        print(result)
        for i in range(0, len(self.states)):
            self.states[i] = (np.array(result).astype(np.byte), np.array(self.states[i]))
        return np.array(self.states)


if __name__ == "__main__":
    for i in range(20):
        s = State(self_play=True)
        s.play()

