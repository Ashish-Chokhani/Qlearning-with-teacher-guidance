#!/usr/bin/env python3
"""
utils.py

"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_rewards(rewards, title):
    """
    Plot cumulative rewards per episode.

    :param rewards: List of rewards
    :param title: Title of the plot
    """
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Episode")
    plt.title(title)
    plt.show()


def plot_heatmap(data, x_labels, y_labels, title, xlabel, ylabel):
    """
    Plot a heatmap of given data with a secondary y-axis label.

    :param data: 2D array containing data to be visualised
    :param x_labels: Labels for the x-axis
    :param y_labels: Labels for the y-axis
    :param title: Title of the heatmap
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    """
    plt.figure(figsize=(8, 6))

    # Create the heatmap
    ax = sns.heatmap(
        data,
        annot=True,
        cmap="coolwarm",
        fmt=".1f",
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    # Set the main title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Invert the y-axis so that the heatmap displays correctly
    ax.invert_yaxis()

    # Create a secondary y-axis for the "Average Reward" label on the right side
    ax2 = ax.twinx()
    ax2.set_ylabel(
        "Average Reward", rotation=90, labelpad=25
    )  # Set the label for the right side
    ax2.set_yticks([])  # Hide the ticks on the right y-axis
    ax2.set_yticklabels([])  # Ensure no tick labels on the right

    # Show the plot
    plt.show()




