from copy import deepcopy
import time
import platform
from io import StringIO

if platform.system() != 'Windows':
    import resource

import sys
import os
import itertools
import numpy as np

sys.path[0] = os.getcwd()

class Board:
    BLANK = " "
    BLOCKED = "X"
    NOT_MOVED = (-1, -1)

    __player_1__ = None
    __player_2__ = None

    __active_player__ = None
    __inactive_player__ = None

    __active_player_name__ = ""
    __inactive_player_name__ = ""

    __active_players_queen1__= None
    __active_players_queen2__= None
    __active_players_queen3__= None
    __inactive_players_queen1__= None
    __inactive_players_queen2__= None
    __inactive_players_queen3__= None

    __last_queen_move__ = {}
    __last_queen_symbols__ = {}
    move_count = 0
    bf_count = 0

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width = width
        self.height = height

        self.__player_1__ = player_1
        self.__player_2__ = player_2

        self.__queen_1_1__ = player_1.__class__.__name__ + " - P1_Q1"
        self.__queen_1_2__ = player_1.__class__.__name__ + " - P1_Q2"
        self.__queen_1_3__ = player_1.__class__.__name__ + " - P1_Q3"
        
        self.__queen_2_1__ = player_2.__class__.__name__ + " - P2_Q1"
        self.__queen_2_2__ = player_2.__class__.__name__ + " - P2_Q2"
        self.__queen_2_3__ = player_2.__class__.__name__ + " - P2_Q3"

        self.__board_state__ = [
            [Board.BLANK for i in range(0, width)] for j in range(0, height)]

        self.__last_queen_move__ = {
            self.__queen_1_1__: Board.NOT_MOVED, self.__queen_2_1__: Board.NOT_MOVED, \
            self.__queen_1_2__: Board.NOT_MOVED, self.__queen_2_2__: Board.NOT_MOVED, \
            self.__queen_1_3__: Board.NOT_MOVED, self.__queen_2_3__: Board.NOT_MOVED}

        self.__queen_symbols__ = {
            Board.BLANK: Board.BLANK,                       \
            self.__queen_1_1__: "11", self.__queen_2_1__: "21", \
            self.__queen_1_2__: "12", self.__queen_2_2__: "22", \
            self.__queen_1_3__: "13", self.__queen_2_3__: "23"}

        self.__active_player__ = player_1
        self.__inactive_player__ = player_2

        self.__active_player_name__ = f"{player_1.__class__.__name__} - Q1"
        self.__inactive_player_name__ = f"{player_2.__class__.__name__} - Q2"

        self.__active_players_queen1__= self.__queen_1_1__
        self.__active_players_queen2__= self.__queen_1_2__
        self.__active_players_queen3__= self.__queen_1_3__
        self.__inactive_players_queen1__= self.__queen_2_1__
        self.__inactive_players_queen2__= self.__queen_2_2__
        self.__inactive_players_queen3__= self.__queen_2_3__

        self.move_count = 0
        self.bf_count = 0

    def get_state(self):
        """
        Get physical board state
        Parameters:
            None
        Returns:
            State of the board: list[char]
        """
        return deepcopy(self.__board_state__)

    def set_state(self, board_state, p1_turn=True):
        '''
        Function to immediately bring a board to a desired state. Useful for testing purposes; call board.play_isolation() afterwards to play.
        Note that error testing is minimal in this function. Please be sure to only pass a list of same size lists of strings. Each string
        should be one of the following: BLANK, BLOCKED, "11", "12", "13", "21", "22", "23".

        Parameters:
            board_state: list[str], Desired state to set to board
            p1_turn: bool, Flag to determine which player is active
        Returns:
            None
        '''
        self.__board_state__ = board_state

        string_options = ["11", "12", "13", "21", "22", "23"]
        all_queens = [self.__queen_1_1__, self.__queen_1_2__, self.__queen_1_3__, self.__queen_2_1__, self.__queen_2_2__, self.__queen_2_3__]

        for queen,string_opt in zip(all_queens,string_options):
            last_move = [(column, row.index(string_opt)) for column, row in enumerate(board_state) if string_opt in row]
            if (last_move != []):
                self.__last_queen_move__[queen] = last_move[0]

        if not p1_turn:
            # rotate the players
            self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__
            self.__active_player_name__, self.__inactive_player_name__ = self.__inactive_player_name__, self.__active_player_name__

            # rotate the queens
            self.__active_players_queen1__,self.__inactive_players_queen1__ = self.__inactive_players_queen1__,self.__active_players_queen1__
            self.__active_players_queen2__,self.__inactive_players_queen2__ = self.__inactive_players_queen2__,self.__active_players_queen2__
            self.__active_players_queen3__,self.__inactive_players_queen3__ = self.__inactive_players_queen3__,self.__active_players_queen3__

        # Count X's to get move count + 6 for initial moves
        self.move_count = sum(row.count('X') + row.count('11') + row.count('12') + row.count('13') + \
                              row.count('21') + row.count('22') + row.count('23') for row in board_state)

    def __apply_move__(self, move):
        '''
        Apply chosen move to a board state and check for game end
        Parameters:
            move: ((int,int),(int,int),(int,int)), Desired move for all 3 queens
        Returns:
            result: (bool, str), Game Over flag, winner
        '''
        queens = self.get_active_players_queens()

        # apply the move, one queen at a time
        for queen_num, queen in enumerate(queens):
            row,col = move[queen_num]
            
            queen_pos = self.__last_queen_move__[queen]
            queen_name = self.__queen_symbols__[queen]
            self.__board_state__[row][col] = queen_name

            if self.move_is_in_board(queen_pos[0], queen_pos[1]):
                self.__board_state__[queen_pos[0]][queen_pos[1]] = Board.BLOCKED

            self.__last_queen_move__[queen] = move[queen_num]
            self.__board_state__[row][col] = self.__queen_symbols__[queen]

        # rotate the players
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__
        self.__active_player_name__, self.__inactive_player_name__ = self.__inactive_player_name__, self.__active_player_name__

        # rotate the queens
        self.__active_players_queen1__,self.__inactive_players_queen1__ = self.__inactive_players_queen1__,self.__active_players_queen1__
        self.__active_players_queen2__,self.__inactive_players_queen2__ = self.__inactive_players_queen2__,self.__active_players_queen2__
        self.__active_players_queen3__,self.__inactive_players_queen3__ = self.__inactive_players_queen3__,self.__active_players_queen3__

        # If opponent is isolated
        if not self.get_active_moves():
            return True, self.__inactive_player_name__

        # increment move count
        self.move_count = self.move_count + 1

        return False, None

    def copy(self):
        '''
        Create a copy of this board and game state.
        Parameters:
            None
        Returns:
            Copy of self: Board class
        '''
        b = Board(self.__player_1__, self.__player_2__,
                  width=self.width, height=self.height)
        for key, value in self.__last_queen_move__.items():
            b.__last_queen_move__[key] = value
        for key, value in self.__queen_symbols__.items():
            b.__queen_symbols__[key] = value

        b.move_count = self.move_count
        b.__active_player__ = self.__active_player__
        b.__active_player_name__ = self.__active_player_name__
        b.__inactive_player__ = self.__inactive_player__
        b.__inactive_player_name__ = self.__inactive_player_name__
        b.__active_players_queen1__ = self.__active_players_queen1__
        b.__active_players_queen2__ = self.__active_players_queen2__
        b.__active_players_queen3__ = self.__active_players_queen3__
        b.__inactive_players_queen1__ = self.__inactive_players_queen1__
        b.__inactive_players_queen2__ = self.__inactive_players_queen2__
        b.__inactive_players_queen3__ = self.__inactive_players_queen3__
        b.__board_state__ = self.get_state()
        return b

    def forecast_move(self, move):
        """
        See what board state would result from making a particular move without changing the board state itself.
        Parameters:
            move: ((int, int),(int, int),(int, int)), Desired move to forecast. Takes the form of
            a (row, column) 3 tuple.

        Returns:
            (Board, bool, str): Resultant board from move, flag for game-over, winner (if game is over)
        """
        new_board = self.copy()
        is_over, winner = new_board.__apply_move__(move)
        return new_board, is_over, winner

    def get_active_player(self):
        """
        See which player is active. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            (Player): the player who's actively taking a turn
        """
        return self.__active_player__

    def get_inactive_player(self):
        """
        See which player is inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            (Player): the player who's waiting for opponent to take a turn
        """
        return self.__inactive_player__

    def get_active_players_queens(self):
        """
        See which queens are active. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            list[str] : List of Queen names of the player who's taking the current turn
        """
        return [self.__active_players_queen1__, self.__active_players_queen2__, self.__active_players_queen3__]

    def get_inactive_players_queens(self):
        """
        See which queens are inactive. Used mostly in play_isolation for display purposes.
        Parameters:
            None
        Returns:
            list[str] : List of Queen names of the player who's waiting for opponent to take a turn
        """
        return [self.__inactive_players_queen1__, self.__inactive_players_queen2__, self.__inactive_players_queen3__]

    def get_inactive_position(self):
        """
        Get position of inactive player (player waiting for opponent to make move) in [(row, column)] format
        Parameters:
            None
        Returns:
           [(int, int),(int, int),(int, int)]: List of (row,col) of inactive players queens
        """
        return [self.__last_queen_move__[self.__inactive_players_queen1__], self.__last_queen_move__[self.__inactive_players_queen2__], self.__last_queen_move__[self.__inactive_players_queen3__]]

    def get_active_position(self):
        """
        Get position of active player (player actively making move) in [(row, column)] format
        Parameters:
            None
        Returns:
           [(int, int),(int, int),(int, int)]: List of (row,col) of active players queens
        """
        return [self.__last_queen_move__[self.__active_players_queen1__], self.__last_queen_move__[self.__active_players_queen2__], self.__last_queen_move__[self.__active_players_queen3__]]

    def get_player_position(self, my_player=None):
        """
        Get position of certain player object. Should pass in yourself to get your position.
        Parameters:
            my_player (Player), Player to get position for
            If calling from within a player class, my_player = self can be passed.
        returns
            [(int, int),(int, int),(int, int)]: List of (row,col) of the players queens

        """
        if my_player == self.__active_player__:
            return self.get_active_position()
        elif my_player == self.__inactive_player__:
            return self.get_inactive_position()
        else:
            raise ValueError("No value for my_player!")

    def get_opponent_position(self, my_player=None):
        """
        Get position of my_player's opponent.
        Parameters:
            my_player (Player), Player to get opponent's position
            If calling from within a player class, my_player = self can be passed.
        returns
            [int, int]: [Row, col] position of my_player's opponent

        """
        if my_player == self.__active_player__:
            return self.get_inactive_position()
        elif my_player == self.__inactive_player__:
            return self.get_active_position()
        else:
            raise ValueError("No value for my_player!")
    
    def get_moves_from_dictionary(self,move_dict,queens):
        
        queen1_moves = np.empty(len(move_dict[queens[0]]),dtype =object)
        queen1_moves[:] = move_dict[queens[0]]

        queen2_moves = np.empty(len(move_dict[queens[1]]),dtype =object)
        queen2_moves[:] = move_dict[queens[1]]

        queen3_moves = np.empty(len(move_dict[queens[2]]),dtype =object)
        queen3_moves[:] = move_dict[queens[2]]

        moves = np.array(np.meshgrid(queen1_moves,queen2_moves,queen3_moves)).T.reshape(-1,3)
        valid_moves = moves[np.logical_not(np.logical_or(moves[:,0] == moves[:,1], np.logical_or(moves[:,0] == moves[:,2], moves[:,1] == moves[:,2])))]
        valid_moves = list(map(tuple,valid_moves))
        
        return valid_moves
    
    def get_inactive_moves(self):
        """
        Get all legal moves of inactive player on current board state as a list of possible moves.
        Parameters:
            move1,move2,move3
        Returns:
           [((int, int),(int, int), (int,int))]: List of all legal moves. Each move takes the form of
            ((row, column), (row, column), (row, column)). Each tuple within the 3-tuple refers to the 
            move by 1st, 2nd, and 3rd queen respectively.
        """
        inactive_queens = self.get_inactive_players_queens()
        move_dict = {inactive_queens[0] : self.__get_moves__(self.__last_queen_move__[inactive_queens[0]]), \
                     inactive_queens[1] : self.__get_moves__(self.__last_queen_move__[inactive_queens[1]]), \
                     inactive_queens[2] : self.__get_moves__(self.__last_queen_move__[inactive_queens[2]])}

        all_moves = self.get_moves_from_dictionary(move_dict,inactive_queens)
        return all_moves

    def get_active_moves(self):
        """
        Get all legal moves of active player on current board state as a list of possible moves.
        Parameters:
            None
        Returns:
           [((int, int),(int, int), (int,int))]: List of all legal moves. Each move takes the form of
            ((row, column), (row, column), (row, column)). Each tuple within the 3-tuple refers to the 
            move by 1st, 2nd, and 3rd queen respectively.
        """
        active_queens = self.get_active_players_queens()
        move_dict = {active_queens[0] : self.__get_moves__(self.__last_queen_move__[active_queens[0]]),
                     active_queens[1] : self.__get_moves__(self.__last_queen_move__[active_queens[1]]),
                     active_queens[2] : self.__get_moves__(self.__last_queen_move__[active_queens[2]])}

        all_moves = self.get_moves_from_dictionary(move_dict,active_queens)
        return all_moves

    def get_player_moves(self, my_player=None):
        """
        Get all legal moves of certain player object. Should pass in yourself to get your moves.
        Parameters:
            my_player (Player), Player to get moves for
            If calling from within a player class, my_player = self can be passed.
        returns
            [((int, int),(int, int), (int,int))]: List of all legal moves. Each move takes the form of
            ((row, column), (row, column), (row, column)). Each tuple within the 3-tuple refers to the 
            move by 1st, 2nd, and 3rd queen respectively.

        """
        if my_player == self.__active_player__:
            return self.get_active_moves()
        elif my_player == self.__inactive_player__:
            return self.get_inactive_moves()
        else:
            raise ValueError("No value for my_player!")

    def get_opponent_moves(self, my_player=None):
        """
        Get all legal moves of the opponent of the player provided. Should pass in yourself to get your opponent's moves.
        If calling from within a player class, my_player = self can be passed.
        Parameters:
            my_player (Player), The player facing the opponent in question
            If calling from within a player class, my_player = self can be passed.
        returns
            [((int, int),(int, int), (int,int))]: List of all legal moves. Each move takes the form of
            ((row, column), (row, column), (row, column)). Each tuple within the 3-tuple refers to the 
            move by 1st, 2nd, and 3rd queen respectively.

        """
        if my_player == self.__active_player__:
            return self.get_inactive_moves()
        elif my_player == self.__inactive_player__:
            return self.get_active_moves()
        else:
            raise ValueError("No value for my_player!")

    def __get_moves__(self, move):
        """
        Get all legal moves of a player on current board state as a list of possible moves. Not meant to be directly called,
        use get_active_moves or get_inactive_moves instead.
        Parameters:
            move: (int, int), Last move made by player in question (where they currently are).
            Takes the form of (row, column).
        Returns:
           [((int, int),(int, int),(int, int))]: List of (row,col) 3-tuples of legal moves
        """

        if move == self.NOT_MOVED:
            return self.get_first_moves()

        r, c = move

        directions = [(-1, 0),(0, -1), (0, 1), (1, 0)]

        moves = []
        dist = 1
        for direction in directions:
            #for dist in range(1, max(self.height, self.width)):
            row = direction[0] * dist + r
            col = direction[1] * dist + c
            if self.move_is_in_board(row, col) and self.is_spot_open(row, col) and (row, col) not in moves:
                moves.append((row, col))

        return moves

    def get_legal_moves_of_queen1(self):
        return self.__get_moves__(self.__last_queen_move__[self.__active_players_queen1__])

    def get_legal_moves_of_queen2(self):
        return self.__get_moves__(self.__last_queen_move__[self.__active_players_queen2__])

    def get_legal_moves_of_queen3(self):
        return self.__get_moves__(self.__last_queen_move__[self.__active_players_queen3__])

    def get_first_moves(self):
        """
        Return all moves for first turn in game (i.e. every board position)
        Parameters:
            None
        Returns:
           [(int, int)]: List of (row,col) tuples of legal moves
        """
        return [(i, j) for i in range(0, self.height)
                for j in range(0, self.width) if self.__board_state__[i][j] == Board.BLANK]

    def move_is_in_board(self, row, col):
        """
        Sanity check for making sure a move is within the bounds of the board.
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] values are within valid ranges
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def is_spot_open(self, row, col):
        """
        Sanity check for making sure a move isn't occupied by an X.
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] position is blank (no X)
        """
        return self.__board_state__[row][col] == Board.BLANK

    def is_spot_queen(self, row, col):
        """
        Sanity check for checking if a spot is occupied by a player
        Parameters:
            row: int, Row position of move in question
            col: int, Column position of move in question
        Returns:
            bool: Whether the [row,col] position is currently occupied by a player's queen
        """
        return self.__board_state__[row][col] in self.__queen_symbols__.values()


    def space_is_open(self, row, col):
        """
        Sanity check to see if a space is within the bounds of the board and blank. Not meant to be called directly if you don't know what
        you're looking for.
        Parameters:
            row: int, Row value of desired space
            col: int, Col value of desired space
        Returns:
            bool: (Row, Col ranges are valid) AND (space is blank)
        """
        return 0 <= row < self.height and \
               0 <= col < self.width and \
               self.__board_state__[row][col] == Board.BLANK

    def print_board(self, legal_moves=[]):
        """
        Function for printing board state & indicating possible moves for active player.
        Parameters:
            legal_moves: [(int, int)], List of legal moves to indicate when printing board spaces.
            Each move takes the form of (row, column).
        Returns:
            Str: Visual interpretation of board state & possible moves for active player
        """

        p1_q1_r, p1_q1_c = self.__last_queen_move__[self.__queen_1_1__]
        p1_q2_r, p1_q2_c = self.__last_queen_move__[self.__queen_1_2__]
        p1_q3_r, p1_q3_c = self.__last_queen_move__[self.__queen_1_3__]

        p2_q1_r, p2_q1_c = self.__last_queen_move__[self.__queen_2_1__]
        p2_q2_r, p2_q2_c = self.__last_queen_move__[self.__queen_2_2__]
        p2_q3_r, p2_q3_c = self.__last_queen_move__[self.__queen_2_3__]

        b = self.__board_state__

        out = '  |'
        for i in range(len(b[0])):
            out += str(i) + ' |'
        out += '\n\r'

        for i in range(len(b)):
            out += str(i) + ' |'
            for j in range(len(b[i])):
                if (i, j) == (p1_q1_r, p1_q1_c):
                    out += self.__queen_symbols__[self.__queen_1_1__]
                elif (i, j) == (p1_q2_r, p1_q2_c):
                    out += self.__queen_symbols__[self.__queen_1_2__]
                elif (i, j) == (p1_q3_r, p1_q3_c):
                    out += self.__queen_symbols__[self.__queen_1_3__]
                elif (i, j) == (p2_q1_r, p2_q1_c):
                    out += self.__queen_symbols__[self.__queen_2_1__]
                elif (i, j) == (p2_q2_r, p2_q2_c):
                    out += self.__queen_symbols__[self.__queen_2_2__]
                elif (i, j) == (p2_q3_r, p2_q3_c):
                    out += self.__queen_symbols__[self.__queen_2_3__]
                elif (i, j) in legal_moves or (i, j) in legal_moves:
                    out += 'o '
                elif b[i][j] == Board.BLANK:
                    out += '  '
                else:
                    out += '><'

                out += '|'
            if i != len(b) - 1:
                out += '\n\r'

        return out

    def play_isolation(self, time_limit=6000, print_moves=False):
        """
        Method to play out a game of isolation with the agents passed into the Board class.
        Initializes and updates move_history variable, enforces timeouts, and prints the game.
        Parameters:
            time_limit: int, time limit in milliseconds that each player has before they time out.
            print_moves: bool, Should the method print details of the game in real time
        Returns:
            (str, [(int, int)], str): Queen of Winner, Move history, Reason for game over.
            Each move in move history takes the form of (row, column).
        """
        move_history = []
        if platform.system() == 'Windows':
            def curr_time_millis():
                return int(round(time.time() * 1000))
        else:
            def curr_time_millis():
                return 1000 * resource.getrusage(resource.RUSAGE_SELF).ru_utime

        while True:
            game_copy = self.copy()
            move_start = curr_time_millis()

            def time_left():
                # print("Limit: "+str(time_limit) +" - "+str(curr_time_millis()-move_start))
                return time_limit - (curr_time_millis() - move_start)

            if print_moves:
                print("\n", self.__active_player_name__, " Turn")

            
            curr_move_queen1, curr_move_queen2, curr_move_queen3 = self.__active_player__.move(game_copy, time_left)
            move = curr_move_queen1, curr_move_queen2, curr_move_queen3
            # Append new move to game history
            if self.__active_player__ == self.__player_1__:
                move_history.append([[move]])
            else:
                move_history[-1].append([move])

            # Handle Timeout
            if time_limit and time_left() <= 0:
                if print_moves:
                    print('Winner: ' + self.__inactive_player_name__)
                return self.__inactive_player_name__, move_history, \
                       self.__active_player_name__ + " timed out."

            # Safety Check
            legal_moves = self.get_active_moves()
            if move not in legal_moves:
                return self.__inactive_player_name__, move_history, \
                       (self.__active_player_name__+ " made an illegal move.")

            # Apply move to game.
            is_over, winner = self.__apply_move__((move))

            if print_moves:
                print("move chosen: Q1 to %s, Q2 to %s, and Q3 to %s" % (curr_move_queen1,curr_move_queen2,curr_move_queen3))
                print(self.copy().print_board())
            if is_over:
                return self.__inactive_player_name__, move_history, self.__active_player_name__ + " has no legal moves left."

    def __apply_move_write__(self, move):
        """
        Equivalent to __apply_move__, meant specifically for applying move history to a board
        for analyzing an already played game.
        Parameters:
            move_queen: (int, int), Move to apply to board. Takes
            the form of (row, column).
        Returns:
            None
        """
        queens = self.get_active_players_queens()

        # apply the move, one queen at a time
        for queen_num, queen in enumerate(queens):
            row,col = move[queen_num]
            
            if move[queen_num][0] is None or move[queen_num][1] is None:
                return
            
            self.__last_queen_move__[queen] = move[queen_num]
            self.__board_state__[row][col] = self.__queen_symbols__[queen]

         # rotate the players
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__
        self.__active_player_name__, self.__inactive_player_name__ = self.__inactive_player_name__, self.__active_player_name__

        # rotate the queens
        self.__active_players_queen1__ = self.__inactive_players_queen1__
        self.__active_players_queen2__ = self.__inactive_players_queen2__
        self.__active_players_queen3__ = self.__inactive_players_queen3__

        self.move_count = self.move_count + 1

def game_as_text(winner, move_history, termination="", board=Board(1, 2)):
    """
    Function to play out a move history on a new board. Used for analyzing an interesting move history
    Parameters:
        move_history: [(int, int)], History of all moves in order of game in question.
        Each move takes the form of (row, column).
        termination: str, Reason for game over of game in question. Obtained from play_isolation
        board: Board, board that game in question was played on. Used to initialize board copy
    Returns:
        Str: Print output of move_history being played out.
    """
    ans = StringIO()

    board = Board(board.__player_1__, board.__player_2__, board.width, board.height)

    print("Printing the game as text.")

    last_move = (9, 9, 0)

    for i, move in enumerate(move_history):
        if move is None or len(move) == 0:
            continue
        
        p1_move = move[0]
        print(move)
        
        if p1_move != Board.NOT_MOVED and p1_move is not None:
            print("Player 1 Turn")
            ans.write(board.print_board())
            board.__apply_move_write__((p1_move[0], p1_move[1], p1_move[2]))
            ans.write("Queen1: (%d,%d) " % p1_move[0] + "Queen2: (%d,%d) " % p1_move[1] + "Queen3: (%d,%d)\r\n" % p1_move[2])
       
        if len(move) > 1 and move[1] != Board.NOT_MOVED and p1_move is not None:
            p2_move = move[1]
            print("Player 2 Turn")
            ans.write(board.print_board())
            board.__apply_move_write__((p2_move[0], p2_move[1], p2_move[2]))
            ans.write("Queen1: (%d,%d) " % p2_move[0] + "Queen2: (%d,%d) " % p2_move[1] + "Queen3: (%d,%d)\r\n" % p2_move[2])

        last_move = move

    ans.write("\n" + str(winner) + " has won. Reason: " + str(termination))
    return ans.getvalue()
