from typing import Dict, List, Optional, Set, Tuple

import gym
import numpy as np
from gym import spaces


class Room:
    """Represents a room in the navigation environment."""

    def __init__(
        self,
        position: Tuple[int, int],
        size: Tuple[int, int],
        doors: Dict[str, bool] = None,
        key_color: Optional[str] = None,
    ):
        """Initialize room.

        Args:
            position: (x, y) position of room's top-left corner
            size: (width, height) of room
            doors: Dictionary of door states (locked/unlocked)
            key_color: Color of key in this room (if any)
        """
        self.position = position
        self.size = size
        self.doors = doors or {}
        self.key_color = key_color

    @property
    def bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get room boundaries.

        Returns:
            Tuple of ((min_x, min_y), (max_x, max_y))
        """
        return (
            self.position,
            (
                self.position[0] + self.size[0],
                self.position[1] + self.size[1],
            ),
        )

    def contains(self, position: Tuple[int, int]) -> bool:
        """Check if position is inside room.

        Args:
            position: Position to check

        Returns:
            Whether position is in room
        """
        (min_x, min_y), (max_x, max_y) = self.bounds
        x, y = position
        return min_x <= x < max_x and min_y <= y < max_y


class NavigationEnv(gym.Env):
    """Multi-room navigation environment with keys and doors."""

    def __init__(
        self,
        num_rooms: int = 9,
        room_size: Tuple[int, int] = (7, 7),
        max_steps: int = 5000,
        num_keys: int = 3,
    ):
        """Initialize navigation environment.

        Args:
            num_rooms: Number of rooms to generate
            room_size: Size of each room
            max_steps: Maximum steps per episode
            num_keys: Number of keys to collect
        """
        super().__init__()

        self.num_rooms = num_rooms
        self.room_size = room_size
        self.max_steps = max_steps
        self.num_keys = num_keys

        # Colors for keys and doors
        self.colors = ["red", "blue", "green", "yellow", "purple"]
        assert num_keys <= len(self.colors), "Too many keys"

        # Calculate grid size based on room layout (3x3 grid of rooms)
        grid_width = int(np.ceil(np.sqrt(num_rooms)))
        self.grid_size = (
            grid_width * (room_size[0] + 1),
            grid_width * (room_size[1] + 1),
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # Move (4) + Use Key
        
        # Observation includes:
        # - Position (2)
        # - Keys collected (num_keys)
        # - Local map view (5x5)
        # - Door states for visible doors
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=0,
                high=max(self.grid_size),
                shape=(2,),
                dtype=np.int32,
            ),
            "keys": spaces.MultiBinary(num_keys),
            "local_map": spaces.Box(
                low=0,
                high=255,
                shape=(5, 5),
                dtype=np.uint8,
            ),
            "door_states": spaces.Box(
                low=0,
                high=1,
                shape=(4,),  # N, S, E, W
                dtype=np.bool_,
            ),
        })

        # Initialize state
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.rooms = []
        self.keys_collected = set()

        # Generate rooms
        self._generate_rooms()

        # Place agent in starting room
        start_room = self.rooms[0]
        self.position = (
            start_room.position[0] + start_room.size[0] // 2,
            start_room.position[1] + start_room.size[1] // 2,
        )

        return self._get_observation()

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute environment step.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        reward = 0
        done = self.current_step >= self.max_steps
        info = {}

        # Handle movement (actions 0-3)
        if action < 4:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dx, dy = directions[action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)

            # Check if movement is valid
            if self._is_valid_move(new_pos):
                self.position = new_pos
                reward -= 0.1  # Small penalty for movement

                # Check if we're collecting a key
                current_room = self._get_room(self.position)
                if (current_room and
                    current_room.key_color and
                    current_room.key_color not in self.keys_collected):
                    self.keys_collected.add(current_room.key_color)
                    reward += 10.0  # Reward for collecting key

        # Handle key usage (action 4)
        elif action == 4:
            current_room = self._get_room(self.position)
            if current_room:
                # Try to unlock adjacent doors
                for direction, locked in current_room.doors.items():
                    if locked:
                        door_color = direction.split("_")[1]  # e.g., "north_red" -> "red"
                        if door_color in self.keys_collected:
                            current_room.doors[direction] = False
                            reward += 5.0  # Reward for unlocking door

        # Check win condition (all keys collected)
        if len(self.keys_collected) == self.num_keys:
            reward += 50.0
            done = True
            info["success"] = True

        return self._get_observation(), reward, done, info

    def _generate_rooms(self) -> None:
        """Generate room layout with doors and keys."""
        grid_width = int(np.ceil(np.sqrt(self.num_rooms)))
        
        # Create rooms in a grid pattern
        for i in range(self.num_rooms):
            grid_x = i % grid_width
            grid_y = i // grid_width
            
            position = (
                grid_x * (self.room_size[0] + 1),
                grid_y * (self.room_size[1] + 1),
            )

            # Add doors with random colors
            doors = {}
            if grid_y > 0:  # North door
                doors[f"north_{np.random.choice(self.colors)}"] = True
            if grid_y < grid_width - 1:  # South door
                doors[f"south_{np.random.choice(self.colors)}"] = True
            if grid_x > 0:  # West door
                doors[f"west_{np.random.choice(self.colors)}"] = True
            if grid_x < grid_width - 1:  # East door
                doors[f"east_{np.random.choice(self.colors)}"] = True

            # Add key (if any)
            key_color = None
            if i > 0 and len(self.colors) > 0:  # Don't put key in starting room
                if np.random.random() < self.num_keys / (self.num_rooms - 1):
                    key_color = self.colors.pop()

            self.rooms.append(Room(position, self.room_size, doors, key_color))

    def _get_room(
        self,
        position: Tuple[int, int],
    ) -> Optional[Room]:
        """Get room containing position.

        Args:
            position: Position to check

        Returns:
            Room containing position, or None
        """
        for room in self.rooms:
            if room.contains(position):
                return room
        return None

    def _is_valid_move(self, position: Tuple[int, int]) -> bool:
        """Check if move to position is valid.

        Args:
            position: Target position

        Returns:
            Whether move is valid
        """
        # Check grid bounds
        if not (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1]):
            return False

        # Get current and target rooms
        current_room = self._get_room(self.position)
        target_room = self._get_room(position)

        # Moving within same room
        if current_room and target_room and current_room == target_room:
            return True

        # Moving between rooms
        if current_room and target_room:
            # Check if we're at a door
            if abs(position[0] - self.position[0]) == 1:  # Horizontal movement
                if position[0] > self.position[0]:  # Moving east
                    door = f"east_{next(iter(self.keys_collected))}"
                    return not current_room.doors.get(door, True)
                else:  # Moving west
                    door = f"west_{next(iter(self.keys_collected))}"
                    return not current_room.doors.get(door, True)
            elif abs(position[1] - self.position[1]) == 1:  # Vertical movement
                if position[1] > self.position[1]:  # Moving south
                    door = f"south_{next(iter(self.keys_collected))}"
                    return not current_room.doors.get(door, True)
                else:  # Moving north
                    door = f"north_{next(iter(self.keys_collected))}"
                    return not current_room.doors.get(door, True)

        return False

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation.

        Returns:
            Dictionary of observations
        """
        # Get current room and local map view
        current_room = self._get_room(self.position)
        local_map = np.zeros((5, 5), dtype=np.uint8)

        if current_room:
            # Fill local map
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x = self.position[0] + dx
                    y = self.position[1] + dy
                    
                    # Mark walls
                    if not self._is_valid_move((x, y)):
                        local_map[dy + 2, dx + 2] = 1
                    
                    # Mark doors
                    room = self._get_room((x, y))
                    if room and room.doors:
                        local_map[dy + 2, dx + 2] = 2
                    
                    # Mark keys
                    if room and room.key_color:
                        local_map[dy + 2, dx + 2] = 3

        # Get door states
        door_states = np.zeros(4, dtype=np.bool_)
        if current_room:
            for i, direction in enumerate(["north", "south", "east", "west"]):
                for door, locked in current_room.doors.items():
                    if door.startswith(direction):
                        door_states[i] = not locked
                        break

        return {
            "position": np.array(self.position),
            "keys": np.array([
                color in self.keys_collected
                for color in self.colors[:self.num_keys]
            ]),
            "local_map": local_map,
            "door_states": door_states,
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional rendered frame
        """
        if mode == "human":
            # Create grid representation
            grid = np.zeros(self.grid_size, dtype=str)
            grid[:] = " "

            # Draw rooms and doors
            for room in self.rooms:
                (min_x, min_y), (max_x, max_y) = room.bounds
                
                # Draw walls
                grid[min_x:max_x, min_y] = "#"
                grid[min_x:max_x, max_y - 1] = "#"
                grid[min_x, min_y:max_y] = "#"
                grid[max_x - 1, min_y:max_y] = "#"

                # Draw doors
                for door, locked in room.doors.items():
                    direction, color = door.split("_")
                    door_char = "D" if locked else "O"
                    
                    if direction == "north":
                        grid[min_x + room.size[0]//2, min_y] = door_char
                    elif direction == "south":
                        grid[min_x + room.size[0]//2, max_y - 1] = door_char
                    elif direction == "east":
                        grid[max_x - 1, min_y + room.size[1]//2] = door_char
                    elif direction == "west":
                        grid[min_x, min_y + room.size[1]//2] = door_char

                # Draw key
                if room.key_color:
                    grid[min_x + room.size[0]//2, min_y + room.size[1]//2] = "K"

            # Draw agent
            grid[self.position[0], self.position[1]] = "P"

            # Print grid
            for y in range(self.grid_size[1]):
                print("".join(grid[:, y]))

            # Print collected keys
            print("\nCollected keys:", self.keys_collected)
            print(f"Steps: {self.current_step}/{self.max_steps}\n")

        return None
