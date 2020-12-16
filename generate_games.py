import numpy as np
from state import State

if __name__ == "__main__":
    num_games = 1000

    states = []

    for i in range(num_games):
        print("game ", i + 1)
        s = State(self_play=True)
        game = s.play()
        for i in range(len(game)):
            states.append(game[i])

    numpy_games = np.array(states)
    print(numpy_games.shape)

    # create and save the array
    f = open("data/10_games.npz", 'wb')
    np.savez(f, numpy_games)
    f.close()
