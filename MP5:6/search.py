# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq
from state import euclidean_distance, MazeState



# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
import heapq

import heapq

def astar(maze):
    """
    Implementation of the A* search algorithm for a maze.

    Args:
        maze: Maze object containing the start state, goal states, and other relevant information.

    Returns:
        A tuple containing the optimal path (a list of tuples from start to goal) and the number of visited states.
        If no path is found, returns (None, number of visited states).
    """
    start_state = maze.get_start()
    goal_states = maze.get_objectives()

    # Priority queue for the frontier (open set)
    open_set = []
    heapq.heappush(open_set, (start_state.dist_from_start + start_state.h, start_state))

    # Dictionary to keep track of visited states and their parent state
    visited_states = {start_state: (None, 0)}

    while open_set:
        _, current_state = heapq.heappop(open_set)
        print(f"current state: {current_state}")

        # Check if the current state is one of the goal states
        if current_state.is_goal():  # Compare current_state.state with the goal tuples
            print(f"Reached goal state: {current_state.state}")
            # Use the backtrack function to reconstruct the path
            return backtrack(visited_states, current_state), len(visited_states)

        # Expand the neighbors
        for neighbor in current_state.get_neighbors():
            if not maze.is_valid_move(current_state.state, neighbor.state):
                continue  # Skip invalid moves

            new_dist = current_state.dist_from_start

            # Check for loops
            if neighbor in visited_states and new_dist >= visited_states[neighbor][1]:
                continue

            # Update visited states and add to the open set
            visited_states[neighbor] = (current_state, new_dist)
            f_value = new_dist + neighbor.h
            heapq.heappush(open_set, (f_value, neighbor))

    # If no path is found, return None
    print("No path found after exploring all possibilities.")
    return None, len(visited_states)

def backtrack(visited_states, goal_state):
    """
    Trace back the path from the goal state to the start.

    Args:
        visited_states: Dictionary with state as the key and (parent_state, dist_from_start) as the value.
        goal_state: The state to start tracing back from.

    Returns:
        A list representing the path from the start state to the goal state.
    """
    path = []
    current_state = goal_state

    while current_state is not None:
        path.append(current_state.state)
        current_state = visited_states[current_state][0]
        print('current state:', current_state)

    path.reverse()  # Reverse to get the path from start to goal
    return path
