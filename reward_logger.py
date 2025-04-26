import matplotlib.pyplot as plt
import json


class RewardLogger:
    def __init__(self, save_path="rewards.json") -> None:
        self.rewards = []
        self.save_path = save_path

    def log(self, reward_value) -> None:
        self.rewards.append(reward_value)

    def save(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self.rewards, f)

    def plot(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards, label="Average Reward per Epoch", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.title("Reward Convergence Over Time")
        plt.legend()
        plt.grid()
        plt.show()
