import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import Neurogym

class NetworkEvaluator:
    def __init__(self, networks, env, num_trials=200, num_repetitions=5):
        self.networks = networks
        self.env = env
        self.num_trials = num_trials
        self.num_repetitions = num_repetitions
        self.accuracies = {i: [] for i in range(len(networks))}
        self.mean_accuracies = {}
        self.std_accuracies = {}

    def run_trials(self):
        for net_idx, network in enumerate(self.networks):
            for rep in range(self.num_repetitions):
                correct_count = 0
                for _ in range(self.num_trials):
                    # Neurogym boilerplate
                    trial_info = self.env.new_trial()
                    ob, gt = self.env.ob, self.env.gt

                    # Convert to numpy, add batch dimension to input
                    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)

                    # Run the network for one trial
                    action_pred, rnn_activity = network(inputs)

                    # Compute performance
                    action_pred = action_pred.detach().numpy()[:, 0, :]
                    choice = np.argmax(action_pred[-1, :])
                    correct = choice == gt[-1]

                    # Record correct prediction
                    if correct:
                        correct_count += 1

                # Calculate accuracy for this repetition
                accuracy = correct_count / self.num_trials
                self.accuracies[net_idx].append(accuracy)

    def calculate_statistics(self):
        self.mean_accuracies = {i: np.mean(self.accuracies[i]) for i in self.accuracies}
        self.std_accuracies = {i: np.std(self.accuracies[i]) for i in self.accuracies}

    def get_statistics(self):
        return self.mean_accuracies, self.std_accuracies

    def plot_results(self, task_name="Task"):
        num_networks = len(self.networks)
        plt.figure(figsize=(10, 6))
        mean_values = [self.mean_accuracies[i] for i in range(num_networks)]
        std_values = [self.std_accuracies[i] for i in range(num_networks)]
        plt.bar(range(num_networks), mean_values, yerr=std_values, capsize=10, color='skyblue', width=0.5)
        #plt.xlabel('Network', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
        plt.title(f'Networks Accuracies across {task_name}', fontsize=16, fontweight='bold')
        plt.xticks(range(num_networks), [f'Network {i+1}' for i in range(num_networks)], fontsize=12)
        plt.ylim(0, 1)  # Assuming accuracy ranges from 0 to 1
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add text annotations for standard deviation
        for i in range(num_networks):
            plt.text(i, mean_values[i] + std_values[i] + 0.02, f'STD: {std_values[i]:.2f}', ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Example usage:
# Assuming net_5_pdm is a list of your 5 networks and env is your environment object
networks = [net_1_pdm, net_2_pdm, net_3_pdm, net_4_pdm, net_5_pdm]  # Replace with actual network instances
env = dataset.env  # Replace with actual environment instance

evaluator = NetworkEvaluator(networks, env)
evaluator.run_trials()
evaluator.calculate_statistics()

# Accessing mean and std accuracies
mean_accuracies_pdm, std_accuracies_pdm = evaluator.get_statistics()
print("Mean Accuracies:", mean_accuracies_pdm)
print("Standard Deviations:", std_accuracies_pdm)

# Plotting results with custom title for a specific task
task_name = "PDM Task"
evaluator.plot_results(task_name=task_name)