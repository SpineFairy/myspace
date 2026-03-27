# %%
"""강화학습 기반 예지제어 Gym 환경 및 PPO 학습 예제.

README 조건을 반영해 Gym 스타일 환경과 Stable-Baselines3 PPO 학습 루프를 제공합니다.
"""

# %%
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - 환경에 따라 gymnasium 미설치 가능
    import gym


# %%
LABELS = [
    "TEMP_RUNAWAY",
    "HUM_UP_TEMP_DOWN",
    "VIB_RISE_TEMP_LAG",
    "VIB_SPIKE_REPEAT",
]
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}

NORMAL_BAND = {
    "temp": (27.0, 33.0),
    "hum": (40.0, 50.0),
    "vib": (2.0, 4.0),
}

DANGER_LIMITS = {
    "temp": (5.0, 75.0),
    "hum": (5.0, 80.0),
    "vib": (0.5, 8.0),
}

ACTION_LEVELS = {
    "cooling": {
        1: (0.01, 0.20),
        2: (0.21, 0.40),
        3: (0.41, 0.60),
    },
    "heater": {
        1: (0.01, 0.20),
        2: (0.21, 0.40),
        3: (0.41, 0.60),
    },
    "dehumidifier": {
        1: (0.01, 0.20),
        2: (0.21, 0.40),
        3: (0.41, 0.60),
    },
    "rail_gear": {
        1: (-0.30, -0.20),
        2: (-0.20, -0.10),
        3: (0.0, 0.0),
        4: (0.01, 0.10),
        5: (0.11, 0.20),
        6: (0.21, 0.30),
    },
}


@dataclass
class EpisodeStatus:
    in_band_streak: int = 0
    entered_band: bool = False
    steps: int = 0


# %%
class PredictiveControlEnv(gym.Env):
    """예지 제어용 Gym 환경.

    Observation: temp, hum, vib, rail gear + 직전 제어값(heater, cooling,
    dehumidifier, rail gear) + 분류 라벨.

    Action: MultiDiscrete([4, 4, 4, 6])
    - cooling(0~3), heater(0~3), dehumidifier(0~3), rail gear(1~6)

    분류 라벨 및 제어 행동:
    - TEMP_RUNAWAY -> 쿨링
    - HUM_UP_TEMP_DOWN -> 제습기, 히터
    - VIB_RISE_TEMP_LAG -> 레일 속도 조절, 쿨링
    - VIB_SPIKE_REPEAT -> 레일 속도 조절 or stop
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps: int = 300, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.MultiDiscrete([4, 4, 4, 6])

        low = np.array([
            DANGER_LIMITS["temp"][0],
            DANGER_LIMITS["hum"][0],
            DANGER_LIMITS["vib"][0],
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ], dtype=np.float32)
        high = np.array([
            DANGER_LIMITS["temp"][1],
            DANGER_LIMITS["hum"][1],
            DANGER_LIMITS["vib"][1],
            6.0,
            3.0,
            3.0,
            3.0,
            6.0,
            float(len(LABELS) - 1),
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.state: Dict[str, float] = {}
        self.prev_action = {
            "cooling": 0,
            "heater": 0,
            "dehumidifier": 0,
            "rail_gear": 3,
        }
        self.label_id = 0
        self.status = EpisodeStatus()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = {
            "temp": self._sample_outside_band(NORMAL_BAND["temp"], DANGER_LIMITS["temp"]),
            "hum": self._sample_outside_band(NORMAL_BAND["hum"], DANGER_LIMITS["hum"]),
            "vib": self._sample_outside_band(NORMAL_BAND["vib"], DANGER_LIMITS["vib"]),
            "rail_gear": float(self.rng.integers(1, 7)),
        }
        self.prev_action = {
            "cooling": 0,
            "heater": 0,
            "dehumidifier": 0,
            "rail_gear": int(self.state["rail_gear"]),
        }
        self.label_id = int(self.rng.integers(0, len(LABELS)))
        self.status = EpisodeStatus()

        obs = self._get_obs()
        info = {"label": LABELS[self.label_id]}
        return obs, info

    def step(self, action: np.ndarray):
        cooling, heater, dehumidifier, rail_gear = self._parse_action(action)
        prev_distance = self._distance_to_normal_band()

        # 상태 업데이트
        self.state["temp"] -= self._apply_level_effect("cooling", cooling, self.state["temp"])
        self.state["temp"] += self._apply_level_effect("heater", heater, self.state["temp"])
        self.state["hum"] -= self._apply_level_effect("dehumidifier", dehumidifier, self.state["hum"])
        self.state["vib"] += self._apply_level_effect("rail_gear", rail_gear, self.state["vib"])
        self.state["rail_gear"] = float(rail_gear)

        self._clip_state()

        self.prev_action = {
            "cooling": cooling,
            "heater": heater,
            "dehumidifier": dehumidifier,
            "rail_gear": rail_gear,
        }

        self.status.steps += 1
        in_band = self._is_in_normal_band()
        if in_band:
            self.status.in_band_streak += 1
            self.status.entered_band = True
        else:
            self.status.in_band_streak = 0

        reward = prev_distance - self._distance_to_normal_band()
        if in_band and self.status.in_band_streak == 1:
            reward += 200.0
        if not in_band and self.status.entered_band and self.status.in_band_streak == 0:
            reward -= 200.0

        terminated = False
        truncated = False
        info = {"label": LABELS[self.label_id]}

        if self._is_over_danger_limit():
            reward -= 400.0
            terminated = True
            info["termination_reason"] = "danger_limit"
        elif self.status.in_band_streak >= 30:
            reward += 500.0
            terminated = True
            info["termination_reason"] = "success"
        elif self.status.steps >= 30 and not self.status.entered_band:
            reward -= 600.0
            terminated = True
            info["termination_reason"] = "failure"
        elif self.status.steps >= self.max_steps:
            truncated = True
            info["termination_reason"] = "timeout"

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int, int, int]:
        cooling = int(action[0])
        heater = int(action[1])
        dehumidifier = int(action[2])
        rail_gear = int(action[3]) + 1
        return cooling, heater, dehumidifier, rail_gear

    def _sample_outside_band(self, band: Tuple[float, float], limits: Tuple[float, float]) -> float:
        low, high = band
        choice = self.rng.choice(["low", "high"])
        if choice == "low":
            value = self.rng.uniform(limits[0], low - 0.5)
        else:
            value = self.rng.uniform(high + 0.5, limits[1])
        return round(value, 3)

    def _apply_level_effect(self, action_name: str, level: int, base_value: float) -> float:
        if level == 0:
            return 0.0
        min_rate, max_rate = ACTION_LEVELS[action_name][level]
        rate = self.rng.uniform(min_rate, max_rate)
        return base_value * rate

    def _distance_to_normal_band(self) -> float:
        distance = 0.0
        for key in ("temp", "hum", "vib"):
            low, high = NORMAL_BAND[key]
            value = self.state[key]
            if value < low:
                distance += low - value
            elif value > high:
                distance += value - high
        return distance

    def _is_in_normal_band(self) -> bool:
        return all(
            NORMAL_BAND[key][0] <= self.state[key] <= NORMAL_BAND[key][1]
            for key in ("temp", "hum", "vib")
        )

    def _is_over_danger_limit(self) -> bool:
        for key in ("temp", "hum", "vib"):
            low, high = DANGER_LIMITS[key]
            value = self.state[key]
            if value < low or value > high:
                return True
        return False

    def _clip_state(self) -> None:
        for key in ("temp", "hum", "vib"):
            low, high = DANGER_LIMITS[key]
            self.state[key] = float(np.clip(self.state[key], low, high))
            self.state[key] = round(self.state[key], 3)

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [
                self.state["temp"],
                self.state["hum"],
                self.state["vib"],
                self.state["rail_gear"],
                float(self.prev_action["cooling"]),
                float(self.prev_action["heater"]),
                float(self.prev_action["dehumidifier"]),
                float(self.prev_action["rail_gear"]),
                float(self.label_id),
            ],
            dtype=np.float32,
        )
        return obs


# %%
if __name__ == "__main__":
    # PPO 학습 예시
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3가 설치되어 있지 않습니다. `pip install stable-baselines3`로 설치하세요."
        ) from exc

    env = PredictiveControlEnv(max_steps=300, seed=42)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"step={env.status.steps:03d} | reward={reward:7.2f} | "
            f"temp={env.state['temp']:.3f} hum={env.state['hum']:.3f} vib={env.state['vib']:.3f} "
            f"label={info.get('label')}"
        )
