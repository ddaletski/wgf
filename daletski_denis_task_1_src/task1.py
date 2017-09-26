import numpy as np


def read_data(players_filename, teams_filename):
    players_count = sum(1 for line in open(players_filename))

    with open(players_filename) as f:
        players_ratings = np.zeros((players_count,), dtype='uint16')
        for i in range(players_count):
            rating = f.readline().split(" ")[1]
            players_ratings[i] = rating


    teams_count = sum(1 for line in open(teams_filename))

    with open(teams_filename) as f:
        teams_ratings = np.zeros((teams_count,), dtype='int32')
        for i in range(teams_count):
            team_players_ratings = map(lambda idx: players_ratings[int(idx)], f.readline().split(" ")[1:])
            teams_ratings[i] = np.sum(list(team_players_ratings))

    return teams_ratings


def cumsum_at(idx, cumsum):
    if idx < 0:
        return 0
    else:
        return cumsum[idx]


def cost_without(idx, cumsum1, cumsum2):
    n = len(cumsum1)
    if idx % 2 == 0:
        left_cost = cumsum_at(idx//2-1, cumsum1)
        right_cost = cumsum_at(n-1, cumsum2) - cumsum_at(idx//2-1, cumsum2)
    else:
        left_cost = cumsum_at(idx//2, cumsum1)
        right_cost = cumsum_at(n-1, cumsum2) - cumsum_at(idx//2-1, cumsum2)

    return left_cost + right_cost


def odd_matching(teams):

    # pairs started from pos. 0 like (teams[0], teams[1]), (teams[2], teams[3])
    pairs_1 = [(teams[i][1], teams[i+1][1]) for i in range(0, len(teams)-2, 2)]
    # pairs started from pos. 1 like (teams[1], teams[2]), (teams[3], teams[4])
    pairs_2 = [(teams[i][1], teams[i+1][1]) for i in range(1, len(teams)-1, 2)]

    # cumulative sum of rating differences of pairs_1
    cumsum_1 = np.cumsum(list(map(lambda pair: pair[1]-pair[0], pairs_1)))
    # cumulative sum of rating differences of pairs_2
    cumsum_2 = np.cumsum(list(map(lambda pair: pair[1]-pair[0], pairs_2)))

    # index of optimal throw-away-team and imbalance cost without this team
    min_cost = (0, cost_without(0, cumsum_1, cumsum_2))
    for i in range(1, len(teams)):
        cost_i = cost_without(i, cumsum_1, cumsum_2) # calculate imbalance without i-th team
        if cost_i < min_cost[1]:
            min_cost = (i, cost_i)

    teams_left = teams[:min_cost[0]] + teams[min_cost[0]+1:] # throw away the team

    # construct pairs of team numbers (w/o thrown team)
    pairs = [(teams_left[i][0], teams_left[i+1][0]) for i in range(0, len(teams_left)-1, 2)]
    return pairs


def even_matching(teams):
    # construct pairs of sibling teams - it is the most balanced matching"
    pairs = [(teams[i][0], teams[i+1][0]) for i in range(0, len(teams)-1, 2)]
    return pairs


def matching(teams_ratings):
    sorted_teams = sorted(enumerate(teams_ratings), key=(lambda team: team[1]))
    if len(teams_ratings) % 2 == 0:
        return even_matching(sorted_teams)
    else:
        return odd_matching(sorted_teams)


if __name__ == "__main__":
    import sys
    import os

    data_path = sys.argv[1]

    tests = ["A", "B", "C", "D"]
    for test in tests:
        players_filename = os.path.join(data_path, "test_" + test, "players.txt")
        teams_filename = os.path.join(data_path, "test_" + test, "teams.txt")
        teams_ratings = read_data(players_filename, teams_filename)

        output_filename = os.path.join("daletski_denis_task_1_team_pairs", "test_" + test + "_pairs.txt")
        with open(output_filename, "w") as f:
            for pair in matching(teams_ratings):
                f.write("%d %d\n" % pair)
