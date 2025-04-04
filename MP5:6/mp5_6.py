# mp5_6.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/04/2018

"""
This file contains the main application that is run for this MP.
"""
import os.path

import numpy as np
import pickle
import pygame
import sys
import argparse
import configparser
import copy

from pygame.locals import *
from alien import Alien
from maze import Maze
from search import search
from const import *
from utils import *
from geometry import *
import time
import traceback


class Application:

    def __init__(self, configfile, map_name, human=True, fps=DEFAULT_FPS):
        self.running = False
        self.displaySurface = None
        self.config = configparser.ConfigParser()
        self.config.read(configfile)
        self.fps = fps
        self.__human = human
        self.clock = pygame.time.Clock()
        self.trajectory = []
        lims = eval(self.config.get(map_name, 'Window'))
        self.alien_limits = [(0, int(lims[0])), (0, int(lims[1]))]
        # Parse config file
        self.windowTitle = "CS440 MP5/6 Shapeshifting Alien"
        self.window = eval(self.config.get(map_name, 'Window'))
        self.centroid = eval(self.config.get(map_name, 'StartPoint'))
        self.widths = eval(self.config.get(map_name, 'Widths'))
        self.alien_shape = 'Ball'
        self.lengths = eval(self.config.get(map_name, 'Lengths'))
        self.alien_shapes = ['Horizontal', 'Ball', 'Vertical']
        self.obstacles = eval(self.config.get(map_name, 'Obstacles'))
        self.goals = eval(self.config.get(map_name, 'Goals'))
        self.waypoints = eval(self.config.get(map_name, 'Waypoints'))

        boundary = [(0, 0, 0, lims[1]), (0, 0, lims[0], 0), (lims[0], 0, lims[0], lims[1]),
                    (0, lims[1], lims[0], lims[1])]
        self.obstacles.extend(boundary)
        self.alien_color = BLACK
        self.alien = Alien(self.centroid, self.lengths, self.widths, self.alien_shapes, self.alien_shape, self.window)

    # Initializes the pygame context and certain properties of the maze
    def initialize(self):

        pygame.init()
        self.font = pygame.font.Font('freesansbold.ttf', 10)
        self.displaySurface = pygame.display.set_mode((self.window[0], self.window[1]), pygame.HWSURFACE)
        self.displaySurface.fill(WHITE)
        pygame.display.flip()
        pygame.display.set_caption(self.windowTitle)
        self.running = True

    def get_alien_color(self):
        if does_alien_touch_wall(self.alien, self.obstacles) or not is_alien_within_window(self.alien, self.window):
            self.alien_color = RED
        else:
            self.alien_color = BLACK

    # Once the application is initiated, execute is in charge of drawing the game and dealing with the game loop
    def execute(self, searchMethod, granularity, trajectory):
        self.granularity = granularity
        self.initialize()
        if not self.running:
            print("Program init failed")
            raise SystemExit

        self.gameLoop()

        if not self.__human:
            maze = Maze(self.alien, self.obstacles, self.waypoints, self.goals, k=5)
            print("Searching the path...")
            try:
                t1 = time.time()
                path = search(maze, searchMethod)
                t2 = time.time()
                print('Time cost: ', t2 - t1)
            except Exception as inst:
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .args
                print(inst)
                print(traceback.format_exc())
                return

            if path is None:
                print("No path found!")
                print('States Explored: ', maze.states_explored)
            else:
                print(path)
                self.trajectory = [x.state for x in path]
                print('Path: ', self.trajectory, 'length: ', len(path))
                print('States Explored: ', maze.states_explored)
                if len(path) > 0:
                    print('Total Cost: ', path[-1].dist_from_start)
                self.gameLoop()
                print("Done!")
                # if saveMaze and not self.__human:
                #     maze.saveToFile(saveMaze)
                self.drawTrajectory(final=True)

        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if (keys[K_ESCAPE]):
                self.running = False

            if self.__human:
                # alpha, beta, gamma = currAngle
                old_config = self.alien.get_centroid()
                x, y = self.alien.get_centroid()
                shape = self.alien_shapes.index(self.alien.get_shape())
                if (keys[K_a]):
                    x -= granularity if isValueInBetween(self.alien_limits[X], x - granularity) else 0
                # move right
                elif (keys[K_d]):
                    x += granularity if isValueInBetween(self.alien_limits[X], x + granularity) else 0
                # move down
                elif (keys[K_s]):
                    y += granularity if isValueInBetween(self.alien_limits[Y], y + granularity) else 0
                # move up
                elif (keys[K_w]):
                    y -= granularity if isValueInBetween(self.alien_limits[Y], y - granularity) else 0
                # changes shape forward
                elif (keys[K_q]):
                    shape = self.alien_shapes[shape + 1] if isValueInBetween([0, len(self.alien_shapes) - 1],
                                                                             shape + 1) else self.alien_shapes[shape]
                    self.alien.set_alien_shape(shape)
                    # debounce
                    time.sleep(0.1)
                # changes shape backward
                elif (keys[K_e]):
                    shape = self.alien_shapes[shape - 1] if isValueInBetween([0, len(self.alien_shapes) - 1],
                                                                             shape - 1) else self.alien_shapes[shape]
                    self.alien.set_alien_shape(shape)
                    # debounce
                    time.sleep(0.1)

                self.alien.set_alien_pos([x, y])
                self.get_alien_color()
                self.gameLoop()

        # if saveMaze and not self.__human:
        #     maze.saveToFile(saveMaze)

    def gameLoop(self):

        self.clock.tick(self.fps)
        self.displaySurface.fill(WHITE)

        # create a text surface object,
        # on which text is drawn on it.
        text = self.font.render('({:.2f},{:.2f})'.format(*self.alien.get_centroid()), True, BLACK, WHITE)

        # create a rectangular object for the
        # text surface object
        textRect = text.get_rect()

        # set the center of the rectangular object.
        textRect.center = (self.window[0] - 40, self.window[1] - 10)

        self.displaySurface.blit(text, textRect)
        # self.drawTrajectory()
        self.drawAlien()
        self.drawObstacles()
        self.drawWayPoints()
        self.drawGoal()
        pygame.display.flip()

    def drawTrajectory(self, final=False):
        cnt = 1
        if final:
            while (True):
                print(self.trajectory)
                for idx, config in enumerate(self.trajectory):
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    if (keys[K_ESCAPE]):
                        pygame.quit()
                        sys.exit()
                    if idx == 0:
                        self.alien.set_alien_config([config[0], config[1], self.alien.get_shapes()[config[2]]])
                        self.get_alien_color()
                        self.gameLoop()
                    else:
                        # interpolation
                        start_x, start_y = self.alien.get_centroid()
                        end_x, end_y = config[0], config[1]
                        interp_x, interp_y = np.linspace(start_x, end_x, 20), np.linspace(start_y, end_y, 20)
                        for x, y in zip(interp_x, interp_y):
                            self.alien.set_alien_config([x, y, self.alien.get_shapes()[config[2]]])
                            self.get_alien_color()
                            self.gameLoop()
                            time.sleep(0.05)
                    time.sleep(0.2)
                time.sleep(3)

    def drawAlien(self):
        self.alien_shape = self.alien.get_shape()
        shape_idx = self.alien_shapes.index(self.alien_shape)
        self.centroid = self.alien.get_centroid()
        if (self.alien_shape == 'Horizontal'):
            head = (self.centroid[0] + self.lengths[shape_idx] / 2, self.centroid[1])
            tail = (self.centroid[0] - self.lengths[shape_idx] / 2, self.centroid[1])
        elif (self.alien_shape == 'Vertical'):
            head = (self.centroid[0], self.centroid[1] + self.lengths[shape_idx] / 2)
            tail = (self.centroid[0], self.centroid[1] - self.lengths[shape_idx] / 2)
        elif (self.alien_shape == 'Ball'):
            head = (self.centroid[0], self.centroid[1])
            tail = (self.centroid[0], self.centroid[1])
        pygame.draw.line(self.displaySurface, self.alien_color, head, tail, self.widths[shape_idx])
        pygame.draw.circle(self.displaySurface, self.alien_color, head, self.widths[shape_idx] / 2)
        pygame.draw.circle(self.displaySurface, self.alien_color, tail, self.widths[shape_idx] / 2)

    def drawObstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.line(self.displaySurface, BLACK, (obstacle[0], obstacle[1]), (obstacle[2], obstacle[3]))

    def drawWayPoints(self):
        for waypoint in self.waypoints:
            pygame.draw.circle(self.displaySurface, GREEN, waypoint, 2)

    def drawGoal(self):
        for goal in self.goals:
            pygame.draw.circle(self.displaySurface, BLUE, (goal[0], goal[1]), 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP5/6 Robotic Arm')

    parser.add_argument('--config', dest="configfile", type=str, default="maps/test_config.txt",
                        help='configuration filename - default Test1')
    parser.add_argument('--map', dest="map_name", type=str, default="Test1",
                        help='configuration filename - default Test1')
    parser.add_argument('--method', dest="search", type=str, default="astar",
                        choices=["astar"],
                        help='search method - default astar')
    parser.add_argument('--human', default=False, action="store_true",
                        help='flag for human playable - default False')
    parser.add_argument('--fps', dest="fps", type=int, default=DEFAULT_FPS,
                        help='fps for the display - default ' + str(DEFAULT_FPS))
    parser.add_argument('--granularity', dest="granularity", type=int, default=DEFAULT_GRANULARITY,
                        help='degree granularity - default ' + str(DEFAULT_GRANULARITY))
    parser.add_argument('--trajectory', dest="trajectory", type=int, default=0,
                        help='leave footprint of rotation trajectory in every x moves - default 0')

    args = parser.parse_args()
    app = Application(args.configfile, args.map_name, args.human, args.fps)
    app.execute(args.search, args.granularity, args.trajectory)
