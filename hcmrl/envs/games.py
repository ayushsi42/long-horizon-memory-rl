from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces


class GameState:
    """Base class for game states."""

    def __init__(self, num_players: int = 2):
        """Initialize game state.

        Args:
            num_players: Number of players
        """
        self.num_players = num_players
        self.current_player = 0

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions.

        Returns:
            List of valid action indices
        """
        raise NotImplementedError

    def apply_action(self, action: int) -> None:
        """Apply action to state.

        Args:
            action: Action to apply
        """
        raise NotImplementedError

    def get_observation(self) -> np.ndarray:
        """Get observation of current state.

        Returns:
            Observation array
        """
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """Check if state is terminal.

        Returns:
            Whether state is terminal
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """Get reward for current state.

        Returns:
            Reward value
        """
        raise NotImplementedError


class ConnectFourState(GameState):
    """Connect Four game state."""

    def __init__(self, board_size: Tuple[int, int] = (6, 7)):
        """Initialize Connect Four state.

        Args:
            board_size: (height, width) of board
        """
        super().__init__(num_players=2)
        self.height, self.width = board_size
        self.board = np.zeros(board_size, dtype=np.int32)
        self.last_move: Optional[Tuple[int, int]] = None

    def get_valid_actions(self) -> List[int]:
        """Get valid columns for dropping pieces.

        Returns:
            List of valid column indices
        """
        return [col for col in range(self.width) if self.board[0, col] == 0]

    def apply_action(self, action: int) -> None:
        """Drop piece in column.

        Args:
            action: Column to drop piece in
        """
        # Find first empty row in column
        for row in range(self.height - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player + 1
                self.last_move = (row, action)
                break

        self.current_player = 1 - self.current_player

    def get_observation(self) -> np.ndarray:
        """Get board state observation.

        Returns:
            Board state array
        """
        return self.board.copy()

    def is_terminal(self) -> bool:
        """Check if game is over.

        Returns:
            Whether game is over
        """
        if not self.last_move:
            return False

        row, col = self.last_move
        player = self.board[row, col]

        # Check horizontal
        count = 0
        for c in range(max(0, col - 3), min(self.width, col + 4)):
            if self.board[row, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        # Check vertical
        count = 0
        for r in range(max(0, row - 3), min(self.height, row + 4)):
            if self.board[r, col] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        # Check diagonal (top-left to bottom-right)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        # Check diagonal (top-right to bottom-left)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col - i
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        # Check if board is full
        return len(self.get_valid_actions()) == 0

    def get_reward(self) -> float:
        """Get reward for current state.

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or ongoing
        """
        if not self.is_terminal():
            return 0.0

        if not self.last_move:
            return 0.0

        row, col = self.last_move
        player = self.board[row, col]

        # Check if last move was winning
        if self._check_win(row, col, player):
            return 1.0 if player == 1 else -1.0

        return 0.0  # Draw

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check if move at (row, col) is winning for player.

        Args:
            row: Row index
            col: Column index
            player: Player number

        Returns:
            Whether move is winning
        """
        # Check horizontal
        count = 0
        for c in range(max(0, col - 3), min(self.width, col + 4)):
            if self.board[row, c] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        # Check vertical
        count = 0
        for r in range(max(0, row - 3), min(self.height, row + 4)):
            if self.board[r, col] == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0

        # Check diagonal (top-left to bottom-right)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        # Check diagonal (top-right to bottom-left)
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col - i
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

        return False


class SequentialGamesEnv(gym.Env):
    """Environment for playing sequential games."""

    def __init__(
        self,
        game_type: str = "connect_four",
        max_steps: int = 2000,
        opponent_type: str = "random",
    ):
        """Initialize games environment.

        Args:
            game_type: Type of game to play
            max_steps: Maximum steps per episode
            opponent_type: Type of opponent
        """
        super().__init__()
        self.game_type = game_type
        self.max_steps = max_steps
        self.opponent_type = opponent_type

        # Initialize game state
        if game_type == "connect_four":
            self.state_class = ConnectFourState
            board_size = (6, 7)
            self.action_space = spaces.Discrete(7)
            self.observation_space = spaces.Box(
                low=0,
                high=2,
                shape=board_size,
                dtype=np.int32,
            )
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.state = self.state_class()
        return self.state.get_observation()

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute environment step.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Check if action is valid
        if action not in self.state.get_valid_actions():
            return (
                self.state.get_observation(),
                -10.0,  # Large penalty for invalid move
                True,
                {"invalid_action": True},
            )

        # Apply player's action
        self.state.apply_action(action)
        
        # Check if game is over
        if self.state.is_terminal():
            return (
                self.state.get_observation(),
                self.state.get_reward(),
                True,
                {"win": self.state.get_reward() > 0},
            )

        # Apply opponent's action
        opponent_action = self._get_opponent_action()
        self.state.apply_action(opponent_action)

        # Get reward and check termination
        reward = self.state.get_reward()
        done = (
            self.state.is_terminal() or
            self.current_step >= self.max_steps
        )

        return (
            self.state.get_observation(),
            reward,
            done,
            {"win": reward > 0} if done else {},
        )

    def _get_opponent_action(self) -> int:
        """Get action for opponent.

        Returns:
            Opponent's action
        """
        valid_actions = self.state.get_valid_actions()

        if self.opponent_type == "random":
            return np.random.choice(valid_actions)
        elif self.opponent_type == "heuristic":
            # Simple heuristic: prefer center columns
            center = self.state.width // 2
            weighted_actions = [
                (action, 1.0 / (abs(action - center) + 1))
                for action in valid_actions
            ]
            total_weight = sum(w for _, w in weighted_actions)
            choice = np.random.random() * total_weight
            
            current_weight = 0
            for action, weight in weighted_actions:
                current_weight += weight
                if current_weight >= choice:
                    return action
            
            return valid_actions[0]  # Fallback
        else:
            raise ValueError(f"Unknown opponent type: {self.opponent_type}")

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional rendered frame
        """
        if mode == "human":
            board = self.state.get_observation()
            
            print("\n  " + " ".join(str(i) for i in range(self.state.width)))
            print(" +" + "-" * (2 * self.state.width - 1) + "+")
            
            for row in range(self.state.height):
                print(f"{row}|", end=" ")
                for col in range(self.state.width):
                    if board[row, col] == 0:
                        print(".", end=" ")
                    elif board[row, col] == 1:
                        print("X", end=" ")
                    else:
                        print("O", end=" ")
                print("|")
            
            print(" +" + "-" * (2 * self.state.width - 1) + "+")
            print(f"\nStep: {self.current_step}/{self.max_steps}")
            print(f"Player to move: {'X' if self.state.current_player == 0 else 'O'}\n")

        return None
