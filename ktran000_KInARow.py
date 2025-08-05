import random
from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Charugundla, Saimanasvi; Tran, Kristina'

class OurAgent(KAgent):
    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Kronos, AI Tactician'
        if twin: self.long_name += ' II'
        self.zobrist_table = {}  # Stores board hashes
        self.zobrist_keys = {}  # Random numbers for hashing
        self.write_count = 0
        self.read_attempts = 0
        self.successful_reads = 0
        self.initialize_zobrist()
        self.move_history = []
        self.playing = None  # Will be set in prepare()

    def initialize_zobrist(self):
        board_size = 7  # Assuming 7x7 board for Five-in-a-Row
        for i in range(board_size):
            for j in range(board_size):
                self.zobrist_keys[(i, j, 'X')] = random.getrandbits(64)
                self.zobrist_keys[(i, j, 'O')] = random.getrandbits(64)

    def compute_hash(self, board):
        h = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                piece = board[i][j]
                if piece in ('X', 'O'):
                    h ^= self.zobrist_keys[(i, j, piece)]
        return h

    def prepare(self, game_type, what_side_to_play, opponent_nickname, 
                expected_time_per_move=0.1, utterances_matter=True):
        """
        Prepares the agent with relevant game details before the match begins.
        """
        self.game_type = game_type
        self.playing = what_side_to_play  # ✅ Set the side to 'X' or 'O'
        self.opponent = opponent_nickname
        self.time_per_move = expected_time_per_move
        self.utterances_matter = utterances_matter

        return "OK"

    def make_move(self, current_state, current_remark, time_limit=1000, 
                  autograding=False, use_alpha_beta=True, use_zobrist_hashing=True, 
                  max_ply=3, special_static_eval_fn=None):
        """
        Uses Minimax with Alpha-Beta Pruning and Zobrist Hashing to determine the best move.
        """
        valid_moves = current_state.get_available_moves()
        best_move = None
        best_value = float('-inf') if self.playing == 'X' else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in valid_moves:
            new_state = current_state.apply_move(move)
            
            # ✅ CALL MINIMAX HERE
            move_value = self.minimax(new_state, max_ply - 1, alpha, beta, maximizing_player=(self.playing == 'X'))
            
            if self.playing == 'X':  # Maximizing player (X)
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
                alpha = max(alpha, best_value)
            else:  # Minimizing player (O)
                if move_value < best_value:
                    best_value = move_value
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:  # Pruning condition
                break

        # Apply the chosen move
        new_state = current_state.apply_move(best_move)
        new_state.change_turn()
        self.move_history.append(best_move)

        # Generate response based on game state
        response = self.generate_response(current_remark)
        return [[best_move, new_state], response]

    def minimax(self, state, depth_remaining, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta pruning and Zobrist Hashing.
        """
        board_hash = self.compute_hash(state.board)

        # Check Zobrist table for stored evaluation
        self.read_attempts += 1
        if board_hash in self.zobrist_table:
            self.successful_reads += 1
            return self.zobrist_table[board_hash]

        # ✅ TERMINAL CONDITION: If depth limit is reached or no moves left
        if depth_remaining == 0 or not state.get_available_moves():
            eval_score = self.static_eval(state)  # ⚠️ Ensure this is called
            self.zobrist_table[board_hash] = eval_score
            self.write_count += 1
            return eval_score

        valid_moves = state.get_available_moves()

        if maximizing_player:  # X's turn (maximize)
            max_eval = float('-inf')
            for move in valid_moves:
                new_state = state.apply_move(move)
                eval_score = self.minimax(new_state, depth_remaining - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:  # Beta cutoff
                    break
            self.zobrist_table[board_hash] = max_eval
            self.write_count += 1
            return max_eval

        else:  # O's turn (minimize)
            min_eval = float('inf')
            for move in valid_moves:
                new_state = state.apply_move(move)
                eval_score = self.minimax(new_state, depth_remaining - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:  # Alpha cutoff
                    break
            self.zobrist_table[board_hash] = min_eval
            self.write_count += 1
            return min_eval


    def generate_response(self, remark):
        if "how you did that" in remark:
            return "I analyzed the board state and selected the move that maximizes my chances of winning based on the current configuration."
        elif "take on the game" in remark:
            return f"The game is progressing well. My strategy is focused on creating multiple winning paths. So far, my prediction is favorable."
        return f"I have made my move. Your turn."

    

    def static_eval(self, state, game_type=None):
        """
        Evaluates the board state and returns a score.
        Higher values favor X (maximizing player), lower values favor O (minimizing player).
        """
        board = state.board
        k = 5  # Default to Five-in-a-Row, but can be changed for different game types

        # If game_type is provided, adjust k accordingly
        if game_type:
            k = game_type.k  

        rows, cols = len(board), len(board[0])
        
        # Scoring Constants
        WIN_SCORE = 10000   # Winning state
        BLOCK_SCORE = 5000  # Blocking opponent's win
        OPEN_FOUR = 1000    # Four in a row with open space
        OPEN_THREE = 500    # Three in a row with open space
        OPEN_TWO = 100      # Two in a row with open space
        
        score = 0

        # Check all rows, columns, and diagonals for patterns
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'X':
                    score += self.evaluate_position(board, i, j, k, 'X', rows, cols)
                elif board[i][j] == 'O':
                    score -= self.evaluate_position(board, i, j, k, 'O', rows, cols)

        return score

    def evaluate_position(self, board, i, j, k, player, rows, cols):
        """
        Helper function to check for win conditions, open fours, threes, and twos.
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Diagonal Down-Right, Diagonal Down-Left
        opponent = 'O' if player == 'X' else 'X'
        
        score = 0

        for di, dj in directions:
            count, open_ends = 1, 0

            # Check forward direction
            x, y = i + di, j + dj
            while 0 <= x < rows and 0 <= y < cols and board[x][y] == player:
                count += 1
                x += di
                y += dj
            if 0 <= x < rows and 0 <= y < cols and board[x][y] == " ":
                open_ends += 1  # Open space on this side

            # Check backward direction
            x, y = i - di, j - dj
            while 0 <= x < rows and 0 <= y < cols and board[x][y] == player:
                count += 1
                x -= di
                y -= dj
            if 0 <= x < rows and 0 <= y < cols and board[x][y] == " ":
                open_ends += 1  # Open space on this side

            # Assign points based on open-ended sequences
            if count >= k:
                return 10000  # Winning move
            elif count == k - 1 and open_ends > 0:
                score += 5000  # One move away from winning
            elif count == k - 2 and open_ends == 2:
                score += 1000  # Open four
            elif count == k - 3 and open_ends == 2:
                score += 500  # Open three
            elif count == k - 4 and open_ends == 2:
                score += 100  # Open two

        return score


    def report_zobrist_stats(self):
        return {
            "Writes": self.write_count,
            "Read Attempts": self.read_attempts,
            "Successful Reads": self.successful_reads
        }

    def introduce(self):
        return (
            "I’m Kronos, an AI designed for strategy and precision."
            " My goal is simple: read the board, adapt, and make the smartest moves."
            " Let’s see how this plays out."
        )

