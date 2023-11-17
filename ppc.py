
#    Project Name: Autonomous Vehicle Control in Unity
#    Description: This C# script is part of the Autonomous Vehicle Control in Unity project. 
#                 The aim of the project is to controls a vehicle from it's start to the selected target destination by avoiding obstacles.
#                 It includes path following, speed adjustment, and data logging. It integrates various components such as
#                 AStar pathfinding, Pure Pursuit Control and PID controller.
#    Author: [Alessandro Assirelli]
#    Date: [Project Course 2022-2023]
#    Version: 1.0

import numpy as np


class PPC :

    def __init__(self):
        self.lastFoundIndex = 0 # Stores the index of the last segment the intersection has been found on
        self.lookAheadDistance = 20; # Distance to look ahead for path intersection

    # Compute the steering angle for path following
    def ComputeSteeringAngle(self, path, currentPosition, currentHeading, L):
        # Parameters:
        # - path: List of waypoints (in {x,z} world coordinates) representing the path the vehicle should follow.
        # - currentPosition: The current position and orientation of the vehicle.
        # - L: The wheelbase of the vehicle (distance between front and rear axles).

        currentGoal = self.ComputeIntersection(path, currentPosition)
        turn_error = self.ComputeTurnError(currentGoal, currentPosition, currentHeading)
        alpha = (np.arctan(2 * np.sin(turn_error) * L / self.lookAheadDistance))
        return alpha

    # Compute the steering angles for both left and right wheels (for vehicles with Ackermann steering)
    # public List<float> ComputeSteeringAngle(List<Vector3> path, Transform currentPosition, float L, float interaxialDistance)
    # {
    #     # Parameters:
    #     # - path: List of waypoints (in {x,z} world coordinates) representing the path the vehicle should follow.
    #     # - currentPosition: The current position and orientation of the vehicle.
    #     # - L: The wheelbase of the vehicle (distance between front and rear axles).
    #     # - interaxialDistance: The distance between left and right wheels.

    #     var currentGoal = ComputeIntersection(path, currentPosition.position);
    #     var turn_error = ComputeTurnError(currentGoal, currentPosition);
    #     var alpha = -(Mathf.Atan(2 * Mathf.Sin(turn_error) * L / lookAheadDistance)) * 180 / Mathf.PI;
    #     var R = lookAheadDistance / (2 * Mathf.Sin(turn_error));
    #     var alpha_left = -(Mathf.Atan(L / (R - interaxialDistance / 2))) * 180 / Mathf.PI;
    #     var alpha_right = -(Mathf.Atan(L / (R + interaxialDistance / 2))) * 180 / Mathf.PI;
    #     List<float> result = new List<float>() { alpha_left, alpha_right };
    #     return result;
    # }

    # Compute the intersection between the vehicle's path and a lookahead distance
    def ComputeIntersection(self, path, currentPos):
        # Parameters:
        # - path: List of waypoints representing the path the vehicle should follow.
        # - currentPos: Current position of the vehicle in world coordinates.

        currentGoal = self.FindGoalPt(path, currentPos)
        return currentGoal

    # Compute the error in turning towards a target position
    def ComputeTurnError(self, currentGoal, currentPosition, currentHeading):
        # Parameters:
        # - currentGoal: The target position the vehicle is trying to reach.
        # - currentPosition: The current position and orientation of the vehicle.

        angle_to_reach = np.arctan2(currentGoal[1] - currentPosition[1], currentGoal[0] - currentPosition[0])
        turn_error = (angle_to_reach - currentHeading)
        return turn_error
    

    # Reset the last found index
    def ResetPPC(self):
        # Resets the index used for finding the last goal point on the path.
        self.lastFoundIndex = 0

    # Find the goal point on the path given the lookahead distance
    def FindGoalPt(self, path, current):
        # Parameters:
        # - path: List of waypoints representing the path the vehicle should follow.
        # - current: Current position of the vehicle in world coordinates.
        # - Ld: Lookahead distance for finding the goal point.
        # - lastFoundIndex: Index of the last found segment on the path.

        goal_pt = np.array([0., 0.])
        nsol = 0
        startingIndex = self.lastFoundIndex
        terminal = False

        for i in range(startingIndex, len(path) - 1):
        
            # Tolerance for checking if a point is on the path
            TOL = 1E-5
            x1 = path[i][0]
            x2 = path[i + 1][0]
            y1 = path[i][1]
            y2 = path[i + 1][1]

            # Maximum and minimum values for x and z coordinates
            maxX = max(x1, x2) + TOL
            minX = min(x1, x2) - TOL
            maxZ = max(y1, y2) + TOL
            minZ = min(y1, y2) - TOL

            x1_center = x1 - current[0]
            x2_center = x2 - current[0]
            y1_center = y1 - current[1]
            y2_center = y2 - current[1]

            dx = x2_center - x1_center
            dy = y2_center - y1_center
            dr = np.sqrt(pow(dx, 2) + pow(dy, 2))
            D = x1_center * y2_center - x2_center * y1_center

            Delta = pow(self.lookAheadDistance, 2) * pow(dr, 2) - pow(D, 2)

            if (Delta >= 0):
                x1_sol = current[0] + (D * dy + self.sign(dy) * dx * np.sqrt(Delta)) / pow(dr, 2)
                x2_sol = current[0] + (D * dy - self.sign(dy) * dx * np.sqrt(Delta)) / pow(dr, 2)
                y1_sol = current[1] + (-D * dx + np.abs(dy) * np.sqrt(Delta)) / pow(dr, 2)
                y2_sol = current[1] + (-D * dx - np.abs(dy) * np.sqrt(Delta)) / pow(dr, 2)

                # Check if the first solution is in range
                if (minX <= x1_sol and x1_sol <= maxX and minZ <= y1_sol and y1_sol <= maxZ) or (minX <= x2_sol and x2_sol <= maxX and minZ <= y2_sol and y2_sol <= maxZ):
                    if (minX <= x1_sol and x1_sol <= maxX and minZ <= y1_sol and y1_sol <= maxZ):

                        goal_pt[0] = x1_sol
                        goal_pt[1] = y1_sol
                        nsol += 1
                    
                    # Check if the second solution is in range
                    if (minX <= x2_sol and x2_sol <= maxX and minZ <= y2_sol and y2_sol <= maxZ):

                        goal_pt[0] = x2_sol
                        goal_pt[1] = y2_sol
                        nsol += 1

                    # If both solutions are valid, select the closest to the second point
                    if (nsol == 2):
                        dr_sol1 = np.sqrt(pow(x2 - x1_sol, 2) + pow(y2 - y1_sol, 2))
                        dr_sol2 = np.sqrt(pow(x2 - x2_sol, 2) + pow(y2 - y2_sol, 2))

                        if (dr_sol1 < dr_sol2):
                        
                            goal_pt[0] = x1_sol
                            goal_pt[1] = y1_sol
                        
                        else:
                        
                            goal_pt[0] = x2_sol
                            goal_pt[1] = y2_sol
                        
                    # Check if the goal point is closer to the current position than the next path point
                    if np.sqrt( (goal_pt[0] - current[0]) ** 2 + (goal_pt[1] - current[1]) ** 2)  <= np.sqrt( (path[i+1, 0] - current[0]) ** 2 + (path[i+1, 1] - current[1]) ** 2):
                    
                        self.lastFoundIndex = i
                        break
                    
                    else:
                    
                        self.lastFoundIndex = i + 1
                        if (self.lastFoundIndex == len(path) - 1):
                        
                            terminal = True
        

        if (nsol == 0):
        
            goal_pt = path[self.lastFoundIndex]

            if (self.lastFoundIndex == len(path) - 2):
            
                goal_pt = path[len(path) - 1]
            
    

        if (terminal or self.lastFoundIndex == len(path) - 1):
        
            goal_pt = path[len(path) - 1]
        
        return goal_pt
    

    def sign(self, x):
        return 1 if x >= 0 else -1
