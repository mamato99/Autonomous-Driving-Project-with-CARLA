# Autonomous Driving Project with CARLA

## Introduction

This project aims to develop software for autonomous driving within the CARLA simulation environment. The software's goal is to enable the vehicle to move autonomously, avoiding pedestrians and other vehicles, adhering to traffic rules, and stopping at red lights or stop signs, all within the limits of road regulations.

### Project Participants
- Amato Mario
- Avitabile Margherita
- Battipaglia Lucia

### Initial Baseline

The initial baseline implements an agent that explores the scene to reach a specific destination. This agent can correctly follow speed limits, traffic lights, and consider nearby vehicles. In addition to these behaviors, the agent can maintain a safe distance from a leading vehicle by monitoring the instantaneous time to collision. However, the agent does not handle intersection behavior, lane changes, and some emergency situations, resulting in an unsatisfactory score.

### Proposed Solution

Our solution aims to minimize violations by implementing lane change functionality, intersection management, stop sign handling, cyclist management, and emergency situation handling.

## Implementation Details

Throughout the project, we will delve into the Operational Design Domain (ODD), the implementation choices stemming from this analysis, the proposed solutions, conducted tests, and obtained results. Finally, we will analyze the limitations of our solution and perform a qualitative analysis of the achieved result.
