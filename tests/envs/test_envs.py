import numpy as np
import pytest
from gym import spaces

from hcmrl.envs.crafting import CraftingEnv, Item
from hcmrl.envs.games import ConnectFourState, SequentialGamesEnv
from hcmrl.envs.navigation import NavigationEnv, Room
from hcmrl.envs.story import Character, Event, StoryEnv


def test_item_initialization():
    """Test item initialization and usage."""
    # Test basic item
    item = Item("wood")
    assert item.name == "wood"
    assert item.durability is None
    assert not item.ingredients

    # Test tool with durability
    pickaxe = Item("stone_pickaxe", durability=128)
    assert pickaxe.durability == 128
    assert pickaxe.current_durability == 128

    # Test item with ingredients
    plank = Item("plank", ingredients={"wood": 1})
    assert plank.ingredients == {"wood": 1}


def test_crafting_env():
    """Test crafting environment."""
    env = CraftingEnv(grid_size=16, max_steps=1000)

    # Test initialization
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Dict)

    # Test reset
    obs = env.reset()
    assert "position" in obs
    assert "inventory" in obs
    assert "grid" in obs
    assert "tools" in obs

    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_room_initialization():
    """Test room initialization."""
    position = (0, 0)
    size = (7, 7)
    doors = {"north_red": True, "south_blue": False}
    key_color = "green"

    room = Room(position, size, doors, key_color)
    assert room.position == position
    assert room.size == size
    assert room.doors == doors
    assert room.key_color == key_color

    # Test bounds calculation
    bounds = room.bounds
    assert bounds == ((0, 0), (7, 7))

    # Test position containment
    assert room.contains((3, 3))
    assert not room.contains((8, 8))


def test_navigation_env():
    """Test navigation environment."""
    env = NavigationEnv(num_rooms=9, room_size=(7, 7), max_steps=1000)

    # Test initialization
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Dict)

    # Test reset
    obs = env.reset()
    assert "position" in obs
    assert "keys" in obs
    assert "local_map" in obs
    assert "door_states" in obs

    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_connect_four_state():
    """Test Connect Four game state."""
    state = ConnectFourState()

    # Test initialization
    assert state.board.shape == (6, 7)
    assert state.current_player == 0

    # Test valid actions
    valid_actions = state.get_valid_actions()
    assert len(valid_actions) == 7  # All columns initially valid

    # Test move application
    action = 3  # Drop in middle column
    state.apply_action(action)
    assert state.board[5, 3] == 1  # Piece at bottom of column
    assert state.current_player == 1  # Player switched

    # Test win detection
    # Create horizontal win
    state.board[5, 0:4] = 1
    assert state.is_terminal()
    assert state.get_reward() == 1.0


def test_sequential_games_env():
    """Test sequential games environment."""
    env = SequentialGamesEnv(game_type="connect_four", max_steps=1000)

    # Test initialization
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Box)

    # Test reset
    obs = env.reset()
    assert obs.shape == (6, 7)

    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_character_initialization():
    """Test character initialization."""
    name = "Hero"
    traits = ["brave", "wise"]
    goals = ["save_kingdom"]
    relationships = {"Villain": -0.5}

    character = Character(name, traits, goals, relationships)
    assert character.name == name
    assert character.traits == traits
    assert character.goals == goals
    assert character.relationships == relationships

    # Test relationship update
    character.update_relationship("Ally", 0.8)
    assert character.relationships["Ally"] == 0.8
    character.update_relationship("Ally", 0.3)
    assert character.relationships["Ally"] == 1.0  # Clamped to 1.0


def test_event_initialization():
    """Test event initialization."""
    event = Event(
        event_type="alliance",
        participants=["Hero", "Ally"],
        effects={"relationship": 0.3},
        preconditions={"relationship": -0.3},
    )
    assert event.event_type == "alliance"
    assert event.participants == ["Hero", "Ally"]
    assert event.effects == {"relationship": 0.3}
    assert event.preconditions == {"relationship": -0.3}


def test_story_env():
    """Test story environment."""
    env = StoryEnv(num_characters=5, max_steps=1000)

    # Test initialization
    assert isinstance(env.action_space, spaces.Dict)
    assert isinstance(env.observation_space, spaces.Dict)

    # Test reset
    obs = env.reset()
    assert "character_states" in obs
    assert "character_traits" in obs
    assert "character_goals" in obs
    assert "story_history" in obs
    assert "coherence_metrics" in obs

    # Test step
    action = {
        "event_type": env.action_space["event_type"].sample(),
        "participants": env.action_space["participants"].sample(),
    }
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


@pytest.mark.parametrize("env_class", [
    CraftingEnv,
    NavigationEnv,
    SequentialGamesEnv,
    StoryEnv,
])
def test_env_interface(env_class):
    """Test environment interface compliance."""
    env = env_class()

    # Test spaces
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert isinstance(env.action_space, spaces.Space)
    assert isinstance(env.observation_space, spaces.Space)

    # Test reset
    obs = env.reset()
    assert env.observation_space.contains(obs)

    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Test render
    rendered = env.render(mode="human")
    assert rendered is None  # Human mode returns None

    # Test close
    env.close()  # Should not raise errors
