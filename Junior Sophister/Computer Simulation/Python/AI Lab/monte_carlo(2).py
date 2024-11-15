# Class definitions for Monte Carlo simulations
import numpy as np
from numpy.random import *

class PiPoints():
    """Object that holds points within a unit square and estimates 
    pi from the fraction lying within a quadrant of the unit circle"""
    
    def __init__(self):
        """Each object contains random points in the unit square"""
        self.pi = 0.0
        self.inside = []
        self.outside = []
        
    def add_points(self, N):
        """Add N points to the object and update the estimate of pi"""

        # Generate random points
        for nn in range(int(N)):
            x,y = rand(), rand()
            # Check if they are inside the unit quadrant
            if (x**2 + y**2) < 1:
                self.inside.append((x,y))
            else:
                self.outside.append((x,y))
                
        # Update the number of points and the estimate of pi
        N_in = len(self.inside)
        N_out = N_in + len(self.outside)
        self.pi = 4*N_in/N_out
    
    def print_info(self):
        """Print the current estimate"""
        print('\nNumber of points = ', len(self.inside) + len(self.outside))
        print('Current estimate = ', self.pi)
        
        
class MontyHall():
    """Simulate attempts at the Monty Hall problem.
    Store the number of successful sticks and changes."""

    def __init__(self):
        self.stick_wins = 0
        self.change_wins = 0
        
    def add_attempts(self, N):
        """Run N attempts of the Monty Hall problem. 
        Store the number of successful sticks and changes"""
        
        for nn in range(int(N)):
            
            # List of doors
            door_list = [i for i in range(3)]
            
            # Select the winning door and the initial choice
            winner = choice(door_list)
            my_choice = choice(door_list)
            
            # Find possible doors that the host could open
            options = door_list.copy()
            options.remove(winner)
            if my_choice in options:
                options.remove(my_choice)
            
            # The host reveals a goat behind one of the doors
            opened = choice(options)
                           
            # Changing strategy: choose the other door
            door_list.remove(my_choice)
            door_list.remove(opened)
            new_choice = door_list[0]
            
            # Check if the strategy worked
            if new_choice == winner:
                self.change_wins += 1
            else:
                self.stick_wins += 1
                
    def print_info(self):
        """Print the current statistics"""
        N = self.change_wins + self.stick_wins
        print('\nTotal number of attempts =', N)
        print('Change wins with probability', self.change_wins/N)
        print('Stick wins with probability', self.stick_wins/N)

