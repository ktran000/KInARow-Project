# KInARow-Project

This project implements a game-playing AI agent in Python for a generalized version of Tic-Tac-Toe called **K-in-a-Row**, completed as part of the University of Washington’s CSE 415: Introduction to Artificial Intelligence.

In this game, players alternate placing Xs and Os on a board of configurable size, with the goal of achieving K consecutive tokens in a row. The game also includes features like forbidden squares and handicaps to increase complexity.

## Agent Features

- Minimax Search with support for alpha-beta pruning
- Static Evaluation Function tailored to game-specific configurations
- Zobrist Hashing for board state reuse and faster evaluation
- Utterance System allowing the agent to participate in in-game dialogue with dynamic, persona-driven comments
- Support for both twin agents and autograder parameters

## Files Included

- `ktran000_KInARow.py`: Main agent file containing the AI logic
- `GameTranscript.pdf`: Example game transcript showing the agent’s behavior in action

## Personal Contributions

I implemented:
- The static evaluation function for different K-in-a-Row configurations
- Zobrist hashing to improve efficiency in repeated state evaluation
- A dialogue system that generated expressive in-game utterances based on state and opponent interaction

## Course Info

- Course: CSE 415 - Introduction to Artificial Intelligence
- Institution: University of Washington
- Term: Winter 2025
