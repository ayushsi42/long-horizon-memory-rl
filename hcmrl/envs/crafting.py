from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces


class Item:
    """Represents a craftable item."""

    def __init__(
        self,
        name: str,
        durability: Optional[int] = None,
        ingredients: Optional[Dict[str, int]] = None,
    ):
        """Initialize item.

        Args:
            name: Item name
            durability: Optional durability for tools
            ingredients: Required ingredients for crafting
        """
        self.name = name
        self.durability = durability
        self.ingredients = ingredients or {}
        self.current_durability = durability

    def use(self) -> bool:
        """Use the item, reducing durability if applicable.

        Returns:
            Whether the item is still usable
        """
        if self.durability is None:
            return True
        
        if self.current_durability > 0:
            self.current_durability -= 1
            return True
        return False

    def repair(self) -> None:
        """Repair the item to full durability."""
        if self.durability is not None:
            self.current_durability = self.durability


class CraftingEnv(gym.Env):
    """Minecraft-inspired crafting environment."""

    def __init__(
        self,
        grid_size: int = 16,
        max_steps: int = 10000,
        resource_density: float = 0.1,
    ):
        """Initialize crafting environment.

        Args:
            grid_size: Size of the grid world
            max_steps: Maximum steps per episode
            resource_density: Density of resources in the world
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.resource_density = resource_density

        # Define items and recipes
        self.items = {
            # Basic resources
            "wood": Item("wood"),
            "stone": Item("stone"),
            "iron_ore": Item("iron_ore"),
            "coal": Item("coal"),

            # Refined materials
            "plank": Item("plank", ingredients={"wood": 1}),
            "stick": Item("stick", ingredients={"plank": 2}),
            "iron_ingot": Item("iron_ingot", ingredients={"iron_ore": 1, "coal": 1}),

            # Tools
            "wooden_pickaxe": Item(
                "wooden_pickaxe",
                durability=32,
                ingredients={"plank": 3, "stick": 2},
            ),
            "stone_pickaxe": Item(
                "stone_pickaxe",
                durability=128,
                ingredients={"stone": 3, "stick": 2},
            ),
            "iron_pickaxe": Item(
                "iron_pickaxe",
                durability=256,
                ingredients={"iron_ingot": 3, "stick": 2},
            ),
        }

        # Define action and observation spaces
        num_actions = 6  # Move (4), Mine, Craft
        num_items = len(self.items)
        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(2,),
                dtype=np.int32,
            ),
            "inventory": spaces.Box(
                low=0,
                high=float("inf"),
                shape=(num_items,),
                dtype=np.int32,
            ),
            "grid": spaces.Box(
                low=0,
                high=num_items,
                shape=(grid_size, grid_size),
                dtype=np.int32,
            ),
            "tools": spaces.Box(
                low=0,
                high=float("inf"),
                shape=(num_items,),
                dtype=np.int32,
            ),
        })

        # Initialize state
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        # Reset step counter
        self.current_step = 0

        # Reset player position
        self.position = np.array([0, 0])

        # Reset inventory and tools
        self.inventory = {item: 0 for item in self.items}
        self.tools = {
            item: Item(name=item, durability=self.items[item].durability)
            for item in self.items
            if self.items[item].durability is not None
        }

        # Generate world grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place resources randomly
        num_resources = int(self.grid_size * self.grid_size * self.resource_density)
        for _ in range(num_resources):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            resource = np.random.choice(["wood", "stone", "iron_ore", "coal"])
            self.grid[x, y] = list(self.items.keys()).index(resource)

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
            new_pos = self.position + np.array([dx, dy])

            # Check bounds
            if (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size):
                self.position = new_pos
                reward -= 0.1  # Small penalty for movement

        # Handle mining (action 4)
        elif action == 4:
            x, y = self.position
            resource_idx = self.grid[x, y]
            
            if resource_idx > 0:  # If there's a resource
                resource = list(self.items.keys())[resource_idx]
                
                # Check if we have the right tool
                required_tool = None
                if resource in ["stone", "iron_ore"]:
                    for tool in ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"]:
                        if self.inventory[tool] > 0:
                            required_tool = tool
                            break

                if required_tool is None and resource in ["stone", "iron_ore"]:
                    reward -= 0.5  # Penalty for mining without proper tool
                else:
                    # Mine the resource
                    self.inventory[resource] += 1
                    self.grid[x, y] = 0
                    reward += 1.0

                    # Use tool durability
                    if required_tool:
                        self.tools[required_tool].use()
                        if self.tools[required_tool].current_durability <= 0:
                            self.inventory[required_tool] -= 1

        # Handle crafting (action 5)
        elif action == 5:
            # Try to craft each item
            for item_name, item in self.items.items():
                if not item.ingredients:  # Skip basic resources
                    continue

                # Check if we have ingredients
                can_craft = True
                for ingredient, amount in item.ingredients.items():
                    if self.inventory[ingredient] < amount:
                        can_craft = False
                        break

                if can_craft:
                    # Consume ingredients
                    for ingredient, amount in item.ingredients.items():
                        self.inventory[ingredient] -= amount

                    # Add crafted item
                    self.inventory[item_name] += 1
                    
                    # Higher reward for more complex items
                    reward += len(item.ingredients) * 2.0

                    # Add tool if applicable
                    if item.durability is not None:
                        self.tools[item_name] = Item(
                            name=item_name,
                            durability=item.durability,
                        )

        # Additional rewards for achieving goals
        if self.inventory["iron_pickaxe"] > 0:
            reward += 10.0
            done = True
            info["success"] = True

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation.

        Returns:
            Dictionary of observations
        """
        return {
            "position": self.position,
            "inventory": np.array([
                self.inventory[item] for item in self.items
            ]),
            "grid": self.grid,
            "tools": np.array([
                self.tools[item].current_durability
                if item in self.tools else 0
                for item in self.items
            ]),
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional rendered frame
        """
        if mode == "human":
            # Print grid
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if x == self.position[0] and y == self.position[1]:
                        print("P", end=" ")
                    elif self.grid[x, y] == 0:
                        print(".", end=" ")
                    else:
                        resource = list(self.items.keys())[self.grid[x, y]]
                        print(resource[0].upper(), end=" ")
                print()

            # Print inventory
            print("\nInventory:")
            for item, count in self.inventory.items():
                if count > 0:
                    print(f"{item}: {count}")

            # Print tools
            print("\nTools:")
            for tool in self.tools:
                if self.inventory[tool] > 0:
                    print(
                        f"{tool}: {self.tools[tool].current_durability}/"
                        f"{self.tools[tool].durability}"
                    )
            print("\n")
        
        return None
