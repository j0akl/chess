import numpy as np
from state import State

def generate_games(num_games, games_location,  net_location=None):

    states = []

    for i in range(num_games):
        if net_location is not None:
            s = State(self_play=True, random=False, net_location=net_location)
        else:
            s = State(self_play=True, random=True)
        game = s.play()
        for state in game:
            states.append(state)

        if i % 10 == 0:
            print("game ", str(i))

    no_draws = list(filter(lambda x: x[0] != 0, states))

    numpy_games = np.array(no_draws) # states)
    print("Done")

    # create and save the array
    # nd in the filepath means no draws included
    f = open(games_location, 'wb')
    np.savez(f, numpy_games)
    f.close()

if __name__ == "__main__":
    generate_games(1000, "data/1k_games_v2.npz", net_location="model/1k_games_v1.pt")

