import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Action labels for readability (used in logs/debugging)
ACTION_NAMES = {
    0: "Do Nothing",
    1: "IV Fluids",
    2: "Antibiotics",
    3: "Oxygen Therapy",
    4: "Vasopressors",
}

# Healthy target ranges for each vital sign
TARGETS = {
    "heart_rate":     (70, 100),
    "blood_pressure": (110, 130),
    "oxygen":         (95, 100),
    "lactate":        (0, 2),
    "infection":      (0, 2),
}

def distance_to_range(value, low, high):
    """
    Returns how far a value is from a healthy range.
    If already inside the range → distance = 0
    """
    if low <= value <= high:
        return 0.0
    return float(min(abs(value - low), abs(value - high)))


class SepsisEnv(gym.Env):
    """
    Custom ICU environment for sepsis treatment.

    The agent's goal:
    Keep all patient vitals within healthy ranges
    using appropriate medical interventions.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.renderer = None

        # 5 discrete treatment actions
        self.action_space = spaces.Discrete(5)

        # Observations are normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # Raw physiological ranges (used for normalization)
        self.raw_low  = np.array([30,  50,  70, 0, 0,   0], dtype=np.float32)
        self.raw_high = np.array([180, 200, 100, 10, 10, 150], dtype=np.float32)

        self.previous_distance = None
        self.reset()

    def normalize(self, values):
        """Scale raw values into [0,1] range"""
        return (values - self.raw_low) / (self.raw_high - self.raw_low + 1e-8)

    def get_observation(self):
        """Return current patient state (normalized)"""
        raw = np.array([
            self.heart_rate,
            self.blood_pressure,
            self.oxygen,
            self.lactate,
            self.infection,
            float(self.time),
        ], dtype=np.float32)

        return np.clip(self.normalize(raw), 0.0, 1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Initialize a new patient with unstable vitals"""
        super().reset(seed=seed)

        self.heart_rate     = float(np.random.uniform(100, 130))
        self.blood_pressure = float(np.random.uniform(75, 100))
        self.oxygen         = float(np.random.uniform(88, 94))
        self.lactate        = float(np.random.uniform(2, 5))
        self.infection      = float(np.random.uniform(4, 7))
        self.time = 0

        self.previous_distance = self.total_distance()
        return self.get_observation(), {}

    def total_distance(self):
        """Sum of distances from all healthy ranges"""
        vitals = [
            (self.heart_rate,     *TARGETS["heart_rate"]),
            (self.blood_pressure, *TARGETS["blood_pressure"]),
            (self.oxygen,         *TARGETS["oxygen"]),
            (self.lactate,        *TARGETS["lactate"]),
            (self.infection,      *TARGETS["infection"]),
        ]

        return sum(distance_to_range(v, lo, hi) for v, lo, hi in vitals)

    def step(self, action):
        """Apply action and simulate one timestep"""
        self.time += 1

        #  Natural disease progression 
        self.heart_rate     += np.random.normal(0.0, 0.5)
        self.blood_pressure -= np.random.normal(0.3, 0.3)
        self.oxygen         -= np.random.normal(0.05, 0.05)
        self.lactate        += np.random.normal(0.05, 0.05)
        self.infection      -= np.random.uniform(0.0, 0.1)

        # Small stabilizing effects 
        if self.oxygen < 93:
            self.oxygen += np.random.uniform(0.0, 0.25)

        if self.blood_pressure > 130:
            excess = self.blood_pressure - 130
            self.blood_pressure -= np.random.uniform(0.5, 1.5) * (1 + excess / 50)

        if self.heart_rate < 60:
            self.heart_rate += np.random.uniform(0.5, 1.5)

        # Prevent unnecessary vasopressors
        if action == 4 and self.blood_pressure > 120:
            action = 0

        #  Treatment effects 
        if action == 0:  # Do nothing
            if self.heart_rate < 65:
                self.heart_rate += np.random.uniform(0.0, 0.8)
            else:
                self.heart_rate -= np.random.uniform(0.0, 0.5)

        elif action == 1:  # IV Fluids
            boost = np.random.uniform(4, 8) if self.blood_pressure < 120 else np.random.uniform(0, 2)
            self.blood_pressure += boost
            self.lactate        -= np.random.uniform(0.3, 0.7)
            self.heart_rate     -= np.random.uniform(0.5, 2.0)

        elif action == 2:  # Antibiotics
            self.infection  -= np.random.uniform(1.0, 2.0)
            self.lactate    -= np.random.uniform(0.2, 0.5)
            self.heart_rate -= np.random.uniform(0.5, 1.5)

        elif action == 3:  # Oxygen
            self.oxygen     += np.random.uniform(3, 6)
            self.heart_rate -= np.random.uniform(0.5, 2.0)

        elif action == 4:  # Vasopressors
            self.blood_pressure += np.random.uniform(6, 12)
            self.heart_rate     += np.random.uniform(1, 3)

        #  Keep values realistic 
        self.heart_rate     = float(np.clip(self.heart_rate,     30, 180))
        self.blood_pressure = float(np.clip(self.blood_pressure, 50, 200))
        self.oxygen         = float(np.clip(self.oxygen,         70, 100))
        self.lactate        = float(np.clip(self.lactate,        0, 10))
        self.infection      = float(np.clip(self.infection,      0, 10))

        #  Reward calculation 
        current_distance = self.total_distance()
        progress = self.previous_distance - current_distance

        reward = (progress * 1.5) - 0.5  # encourage improvement, penalize time

        # Bonus for each vital in range
        in_range = [
            TARGETS["heart_rate"][0]     <= self.heart_rate     <= TARGETS["heart_rate"][1],
            TARGETS["blood_pressure"][0] <= self.blood_pressure <= TARGETS["blood_pressure"][1],
            TARGETS["oxygen"][0]         <= self.oxygen         <= TARGETS["oxygen"][1],
            TARGETS["lactate"][0]        <= self.lactate        <= TARGETS["lactate"][1],
            TARGETS["infection"][0]      <= self.infection      <= TARGETS["infection"][1],
        ]

        reward += sum(in_range) * 2.0

        # Oxygen penalty (continuous)
        if self.oxygen < 95:
            reward -= (95 - self.oxygen) * 0.4

        # Safety penalties
        if self.blood_pressure < 75:
            reward -= (75 - self.blood_pressure) * 0.3
        if self.blood_pressure > 140:
            reward -= (self.blood_pressure - 140) * 0.2
        if self.heart_rate < 60:
            reward -= (60 - self.heart_rate) * 0.3

        self.previous_distance = current_distance

        # Termination conditions 
        terminated = False
        truncated  = False
        recovered  = False
        death      = False

        if self.blood_pressure < 60 or self.oxygen < 80 or self.lactate > 9:
            reward -= 100
            terminated = True
            death = True

        elif all(in_range):
            reward += 200
            terminated = True
            recovered = True

        if self.time >= 150:
            truncated = True

        info = {
            "heart_rate":     round(self.heart_rate, 1),
            "blood_pressure": round(self.blood_pressure, 1),
            "oxygen":         round(self.oxygen, 1),
            "lactate":        round(self.lactate, 2),
            "infection":      round(self.infection, 2),
            "time":           self.time,
            "action_name":    ACTION_NAMES[int(action)],
            "in_range_count": sum(in_range),
            "total_distance": round(current_distance, 2),
            "recovered":      recovered,
            "death":          death,
        }

        if self.render_mode == "human":
            self.render()

        return self.get_observation(), reward, terminated, truncated, info

    def render(self):
        """Optional visual ICU display"""
        if self.render_mode == "human":
            from environment.rendering import ICURenderer
            if self.renderer is None:
                self.renderer = ICURenderer()

            self.renderer.draw(
                hr=self.heart_rate,
                bp=self.blood_pressure,
                o2=self.oxygen,
                lac=self.lactate,
                inf=self.infection,
                t=self.time,
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None