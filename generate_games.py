import numpy as np
from state import State

def generate_games(num_games, games_location,  net_location=None):

    states = []

    if net_location is not None:
        s = State(self_play=True, random=False, net_location=net_location)
    else:
        s = State(self_play=True, random=True)

    for i in range(num_games):
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
    generate_games(1000, "data/1k_v2.npz", "model/10k_v1.pt")

