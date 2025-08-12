from typing import Dict, List, Optional, Set, Tuple

import gym
import numpy as np
from gym import spaces


class Character:
    """Represents a character in the story."""

    def __init__(
        self,
        name: str,
        traits: List[str],
        goals: List[str],
        relationships: Dict[str, float] = None,
    ):
        """Initialize character.

        Args:
            name: Character name
            traits: List of character traits
            goals: List of character goals
            relationships: Dictionary of relationships with other characters
        """
        self.name = name
        self.traits = traits
        self.goals = goals
        self.relationships = relationships or {}
        self.state = {}  # Current state/emotions

    def update_relationship(self, other: str, delta: float) -> None:
        """Update relationship with another character.

        Args:
            other: Other character's name
            delta: Change in relationship value
        """
        self.relationships[other] = np.clip(
            self.relationships.get(other, 0.0) + delta,
            -1.0,
            1.0,
        )


class Event:
    """Represents a story event."""

    def __init__(
        self,
        event_type: str,
        participants: List[str],
        effects: Dict[str, float],
        preconditions: Dict[str, float] = None,
    ):
        """Initialize event.

        Args:
            event_type: Type of event
            participants: List of participating characters
            effects: Effects on character relationships
            preconditions: Required relationship values
        """
        self.event_type = event_type
        self.participants = participants
        self.effects = effects
        self.preconditions = preconditions or {}


class StoryEnv(gym.Env):
    """Procedural story generation environment."""

    def __init__(
        self,
        num_characters: int = 5,
        max_steps: int = 3000,
        coherence_threshold: float = 0.6,
    ):
        """Initialize story environment.

        Args:
            num_characters: Number of characters in story
            max_steps: Maximum steps per episode
            coherence_threshold: Minimum coherence for success
        """
        super().__init__()
        self.num_characters = num_characters
        self.max_steps = max_steps
        self.coherence_threshold = coherence_threshold

        # Define character traits and goals
        self.possible_traits = [
            "brave", "wise", "kind", "ambitious", "loyal",
            "deceitful", "vengeful", "curious", "protective", "reckless",
        ]
        self.possible_goals = [
            "find_treasure", "save_kingdom", "gain_power", "find_love",
            "seek_revenge", "protect_family", "discover_truth", "achieve_peace",
        ]

        # Define event types
        self.event_types = {
            "alliance": {
                "participants": 2,
                "effects": {"relationship": 0.3},
                "preconditions": {"relationship": -0.3},
            },
            "betrayal": {
                "participants": 2,
                "effects": {"relationship": -0.5},
                "preconditions": {"relationship": 0.3},
            },
            "conflict": {
                "participants": 2,
                "effects": {"relationship": -0.2},
                "preconditions": {},
            },
            "assistance": {
                "participants": 2,
                "effects": {"relationship": 0.2},
                "preconditions": {"relationship": -0.2},
            },
            "revelation": {
                "participants": 2,
                "effects": {"relationship": 0.1},
                "preconditions": {},
            },
        }

        # Define action and observation spaces
        num_event_types = len(self.event_types)
        self.action_space = spaces.Dict({
            "event_type": spaces.Discrete(num_event_types),
            "participants": spaces.MultiDiscrete([num_characters] * 2),
        })

        # Observation includes:
        # - Character states (traits, goals, relationships)
        # - Story history (last N events)
        # - Current coherence metrics
        self.observation_space = spaces.Dict({
            "character_states": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(num_characters, num_characters),
                dtype=np.float32,
            ),
            "character_traits": spaces.MultiBinary(
                (num_characters, len(self.possible_traits))
            ),
            "character_goals": spaces.MultiBinary(
                (num_characters, len(self.possible_goals))
            ),
            "story_history": spaces.Box(
                low=0,
                high=num_event_types,
                shape=(10, 3),  # Last 10 events: [type, char1, char2]
                dtype=np.int32,
            ),
            "coherence_metrics": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3,),  # [relationship, goal, trait coherence]
                dtype=np.float32,
            ),
        })

        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self.characters = []
        self.story_events = []

        # Generate characters
        for i in range(self.num_characters):
            # Select random traits and goals
            num_traits = np.random.randint(2, 4)
            num_goals = np.random.randint(1, 3)
            
            traits = np.random.choice(
                self.possible_traits,
                size=num_traits,
                replace=False,
            ).tolist()
            goals = np.random.choice(
                self.possible_goals,
                size=num_goals,
                replace=False,
            ).tolist()

            # Initialize character
            character = Character(
                name=f"Character_{i}",
                traits=traits,
                goals=goals,
            )

            # Initialize relationships
            for other in self.characters:
                initial_relationship = np.random.uniform(-0.2, 0.2)
                character.relationships[other.name] = initial_relationship
                other.relationships[character.name] = initial_relationship

            self.characters.append(character)

        return self._get_observation()

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute environment step.

        Args:
            action: Dictionary with event_type and participants

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Parse action
        event_type = list(self.event_types.keys())[action["event_type"]]
        participants = [
            self.characters[i].name for i in action["participants"]
        ]

        # Check if action is valid
        if not self._is_valid_event(event_type, participants):
            return (
                self._get_observation(),
                -1.0,  # Penalty for invalid event
                True,
                {"invalid_event": True},
            )

        # Create and apply event
        event = Event(
            event_type=event_type,
            participants=participants,
            effects=self.event_types[event_type]["effects"].copy(),
        )
        self._apply_event(event)

        # Calculate reward based on coherence
        coherence = self._calculate_coherence()
        reward = coherence.mean()

        # Check termination
        done = (
            self.current_step >= self.max_steps or
            coherence.mean() >= self.coherence_threshold
        )

        return (
            self._get_observation(),
            reward,
            done,
            {
                "coherence": coherence,
                "success": coherence.mean() >= self.coherence_threshold,
            },
        )

    def _is_valid_event(self, event_type: str, participants: List[str]) -> bool:
        """Check if event is valid.

        Args:
            event_type: Type of event
            participants: List of participating characters

        Returns:
            Whether event is valid
        """
        # Check number of participants
        if len(participants) != self.event_types[event_type]["participants"]:
            return False

        # Check for duplicate participants
        if len(set(participants)) != len(participants):
            return False

        # Check preconditions
        preconditions = self.event_types[event_type]["preconditions"]
        for p1 in participants:
            char1 = next(c for c in self.characters if c.name == p1)
            for p2 in participants:
                if p1 != p2:
                    relationship = char1.relationships.get(p2, 0.0)
                    for metric, threshold in preconditions.items():
                        if metric == "relationship" and relationship < threshold:
                            return False

        return True

    def _apply_event(self, event: Event) -> None:
        """Apply event effects to story state.

        Args:
            event: Event to apply
        """
        # Update relationships
        for p1 in event.participants:
            char1 = next(c for c in self.characters if c.name == p1)
            for p2 in event.participants:
                if p1 != p2:
                    char1.update_relationship(
                        p2,
                        event.effects["relationship"],
                    )

        # Add event to story
        self.story_events.append(event)

    def _calculate_coherence(self) -> np.ndarray:
        """Calculate story coherence metrics.

        Returns:
            Array of coherence metrics
        """
        if not self.story_events:
            return np.zeros(3)

        # Relationship coherence
        relationship_changes = []
        for event in self.story_events:
            for p1 in event.participants:
                char1 = next(c for c in self.characters if c.name == p1)
                for p2 in event.participants:
                    if p1 != p2:
                        relationship_changes.append(
                            abs(event.effects["relationship"])
                        )
        relationship_coherence = (
            np.mean(relationship_changes)
            if relationship_changes
            else 0.0
        )

        # Goal coherence
        goal_progress = []
        for char in self.characters:
            goal_events = 0
            for event in self.story_events:
                if char.name in event.participants:
                    # Simple heuristic: events contribute to goals
                    goal_events += 1
            goal_progress.append(min(1.0, goal_events / 10))
        goal_coherence = np.mean(goal_progress)

        # Trait coherence
        trait_consistency = []
        for char in self.characters:
            trait_events = 0
            for event in self.story_events:
                if char.name in event.participants:
                    # Check if event aligns with traits
                    if "brave" in char.traits and event.event_type in ["conflict", "assistance"]:
                        trait_events += 1
                    elif "kind" in char.traits and event.event_type in ["assistance", "alliance"]:
                        trait_events += 1
                    elif "deceitful" in char.traits and event.event_type == "betrayal":
                        trait_events += 1
            trait_consistency.append(min(1.0, trait_events / 5))
        trait_coherence = np.mean(trait_consistency)

        return np.array([
            relationship_coherence,
            goal_coherence,
            trait_coherence,
        ])

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation.

        Returns:
            Dictionary of observations
        """
        # Character states (relationship matrix)
        character_states = np.zeros(
            (self.num_characters, self.num_characters),
            dtype=np.float32,
        )
        for i, char1 in enumerate(self.characters):
            for j, char2 in enumerate(self.characters):
                if i != j:
                    character_states[i, j] = char1.relationships.get(
                        char2.name,
                        0.0,
                    )

        # Character traits
        character_traits = np.zeros(
            (self.num_characters, len(self.possible_traits)),
            dtype=np.int32,
        )
        for i, char in enumerate(self.characters):
            for trait in char.traits:
                j = self.possible_traits.index(trait)
                character_traits[i, j] = 1

        # Character goals
        character_goals = np.zeros(
            (self.num_characters, len(self.possible_goals)),
            dtype=np.int32,
        )
        for i, char in enumerate(self.characters):
            for goal in char.goals:
                j = self.possible_goals.index(goal)
                character_goals[i, j] = 1

        # Story history
        story_history = np.zeros((10, 3), dtype=np.int32)
        for i, event in enumerate(self.story_events[-10:]):
            story_history[i, 0] = list(self.event_types.keys()).index(
                event.event_type
            )
            for j, participant in enumerate(event.participants):
                story_history[i, j + 1] = next(
                    i for i, c in enumerate(self.characters)
                    if c.name == participant
                )

        # Coherence metrics
        coherence_metrics = self._calculate_coherence()

        return {
            "character_states": character_states,
            "character_traits": character_traits,
            "character_goals": character_goals,
            "story_history": story_history,
            "coherence_metrics": coherence_metrics,
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment.

        Args:
            mode: Rendering mode

        Returns:
            Optional rendered frame
        """
        if mode == "human":
            print("\n=== Story State ===")
            
            # Print characters
            print("\nCharacters:")
            for char in self.characters:
                print(f"\n{char.name}:")
                print(f"  Traits: {', '.join(char.traits)}")
                print(f"  Goals: {', '.join(char.goals)}")
                print("  Relationships:")
                for other, value in char.relationships.items():
                    print(f"    {other}: {value:.2f}")

            # Print recent events
            print("\nRecent Events:")
            for event in self.story_events[-5:]:
                print(
                    f"\n{event.event_type.title()}:"
                    f" {' and '.join(event.participants)}"
                )

            # Print coherence metrics
            coherence = self._calculate_coherence()
            print("\nCoherence Metrics:")
            print(f"  Relationship: {coherence[0]:.2f}")
            print(f"  Goal: {coherence[1]:.2f}")
            print(f"  Trait: {coherence[2]:.2f}")
            print(f"  Overall: {coherence.mean():.2f}")

            print(f"\nStep: {self.current_step}/{self.max_steps}\n")

        return None
