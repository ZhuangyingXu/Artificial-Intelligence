#!/usr/bin/env python
import traceback
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer, Player
import platform
import random


if platform.system() != 'Windows':
    import resource

from time import time, sleep

def correctOpenEvalFn(yourOpenEvalFn):
    print()
    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            ["11", " ", " ", "22", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["12", " ", " ", "21", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " "],
            ["13", " ", " ", " ", " ", " ", "23"]
        ]
        sample_board.set_state(board_state, True)
        #test = sample_board.get_legal_moves()
        h = yourOpenEvalFn()
        print('OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board, sample_board.get_active_player())))
    except NotImplementedError:
        print('OpenMoveEvalFn Test: Not implemented')
    except:
        print('OpenMoveEvalFn Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()

def beatRandom(yourAgent):

    """Example test you can run
    to make sure your AI does better
    than random."""

    print("")
    try:
        r = RandomPlayer()
        p = yourAgent()
        game = Board(r, p, 7, 7)
        output_b = game.copy()
        # assign a random move to each player before playing
        for idx in range(2):
            moves = game.get_active_moves()
            random.shuffle(moves)
            move = moves[0]
            game, _, _ = game.forecast_move(move)
        winner, move_history, termination = game.play_isolation(time_limit=6000, print_moves=True)
        print("\n", winner, " has won. Reason: ", termination)
        # Uncomment to see game
        # print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print('CustomPlayer Test: Not Implemented')
    except:
        print('CustomPlayer Test: ERROR OCCURRED')
        print(traceback.format_exc())

    print()


def algorithmTest(yourAgent, algorithm, algorithm_name):
    """Example test to make sure
        your algorithm works, using the
        OpenMoveEvalFunction evaluation function.
        This can be used for debugging your code
        with different model Board states.
        Especially important to check alphabeta
        pruning"""

    # create dummy 5x5 board
    print(f"Running the {algorithm_name} test")
    print()
    try:
        def time_left():  # For these testing purposes, let's ignore timeouts
            return 10000

        player = yourAgent()  # using as a dummy player to create a board
        sample_board = Board(player, RandomPlayer())
        # setting up the board as though we've been playing
        board_state = [
            ["X", "X", "12", " ", "13", "X", " "],
            ["X", "X", " ", " ", " ", " ", "X"],
            ["X", " ", "11", " ", " ", " ", "X"],
            ["X", " ", " ", "X", " ", "X", " "],
            [" ", " ", "22", " ", " ", " ", " "],
            [" ", " ", " ", " ", "21", " ", "X"],
            ["X", "23", " ", " ", "X", " ", "X"]
        ]
        sample_board.set_state(board_state, p1_turn=True)

        test_pass = True

        expected_depth_scores = [(1, -16), (2, -16), (3, -7), (4, 1)]

        for depth, exp_score in expected_depth_scores:
            move, score = algorithm(player, sample_board, time_left, depth=depth, my_turn=True)
            print(score)
            if exp_score != score:
                print(f"{algorithm_name} failed for depth: ", depth)
                test_pass = False
            else:
                print(f"{algorithm_name} passed for depth: ", depth)

        if test_pass:
            player = yourAgent()
            sample_board = Board(RandomPlayer(), player)
            # setting up the board as though we've been playing
            board_state = [
                ["X", "X", "22", " ", " ", "X", " "],
                [" ", "", " ", " ", " ", " ", "X"],
                [" ", "11", " ", " ", " ", " ", " "],
                [" ", "X", " ", "X", " ", " ", " "],
                ["12", " ", " ", "21", " ", "13", "X"],
                [" ", " ", " ", " ", " ", "X", " "],
                ["X", "23", " ", " ", "", "X", "X"]
            ]
            sample_board.set_state(board_state, p1_turn=False)

            test_pass = True

            expected_depth_scores = [(1, 6), (2, 5), (3, 5), (4, 2)]

            for depth, exp_score in expected_depth_scores:
                move, score = algorithm(player, sample_board, time_left, depth=depth, my_turn=True)
                print(score)
                if exp_score != score:
                    print(f"{algorithm_name} failed for depth: ", depth)
                    test_pass = False
                else:
                    print(f"{algorithm_name} passed for depth: ", depth)

        if test_pass:
            print(f"{algorithm_name} Test: Runs Successfully!")

        else:
            print(f"{algorithm_name} Test: Failed")

    except NotImplementedError:
        print(f'{algorithm_name} Test: Not implemented')
    except:
        print(f'{algorithm_name} Test: ERROR OCCURRED')
        print(traceback.format_exc())

