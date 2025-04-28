import json


class RewardLogger:
    def __init__(self) -> None:
        self.rewards = []

    def log(self, reward_value) -> None:
        self.rewards.append(reward_value)

    def save(self, save_path: str = "rewards.json") -> None:
        with open(save_path, "w") as f:
            json.dump(self.rewards, f)
