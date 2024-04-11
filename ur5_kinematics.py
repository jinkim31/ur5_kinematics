import numpy as np


def tf0to1(joints):
    return np.array([
        [np.cos(joints[0]), -np.sin(joints[0]), 0, 0],
        [np.sin(joints[0]), np.cos(joints[0]), 0, 0],
        [0, 0, 1, 89 / 1000],
        [0, 0, 0, 1]])


def tf0to2(joints):
    return np.array([
        [np.cos(joints[0]) * np.cos(joints[1]), -np.cos(joints[0]) * np.sin(joints[1]), -np.sin(joints[0]),
         -(109 * np.sin(joints[0])) / 1000],
        [np.cos(joints[1]) * np.sin(joints[0]), -np.sin(joints[0]) * np.sin(joints[1]), np.cos(joints[0]),
         (109 * np.cos(joints[0])) / 1000],
        [-np.sin(joints[1]), -np.cos(joints[1]), 0, 89 / 1000],
        [0, 0, 0, 1]])


def tf0to3(joints):
    return np.array([
        [np.cos(joints[1] + joints[2]) * np.cos(joints[0]), -np.sin(joints[1] + joints[2]) * np.cos(joints[0]),
         -np.sin(joints[0]),
         (17 * np.cos(joints[0]) * np.cos(joints[1])) / 40 - (109 * np.sin(joints[0])) / 1000],
        [np.cos(joints[1] + joints[2]) * np.sin(joints[0]), -np.sin(joints[1] + joints[2]) * np.sin(joints[0]),
         np.cos(joints[0]),
         (109 * np.cos(joints[0])) / 1000 + (17 * np.cos(joints[1]) * np.sin(joints[0])) / 40],
        [-np.sin(joints[1] + joints[2]), -np.cos(joints[1] + joints[2]), 0, 89 / 1000 - (17 * np.sin(joints[1])) / 40],
        [0, 0, 0, 1]])


def tf0to4(joints):
    return np.array([
        [np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]),
         -np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]), -np.sin(joints[0]),
         (17 * np.cos(joints[0]) * np.cos(joints[1])) / 40 - (109 * np.sin(joints[0])) / 1000 + (
                 49 * np.cos(joints[1] + joints[2]) * np.cos(joints[0])) / 125],
        [np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]),
         -np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]), np.cos(joints[0]),
         (109 * np.cos(joints[0])) / 1000 + (17 * np.cos(joints[1]) * np.sin(joints[0])) / 40 + (
                 49 * np.cos(joints[1] + joints[2]) * np.sin(joints[0])) / 125],
        [-np.sin(joints[1] + joints[2] + joints[3]), -np.cos(joints[1] + joints[2] + joints[3]), 0,
         89 / 1000 - (17 * np.sin(joints[1])) / 40 - (49 * np.sin(joints[1] + joints[2])) / 125],
        [0, 0, 0, 1]])


def tf0to5(joints):
    return np.array([
        [np.sin(joints[0]) * np.sin(joints[4]) + np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.cos(
            joints[4]),
         np.cos(joints[4]) * np.sin(joints[0]) - np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.sin(
             joints[4]),
         -np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]),
         (17 * np.cos(joints[0]) * np.cos(joints[1])) / 40 - (109 * np.sin(joints[0])) / 1000 - (
                 19 * np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[0])) / 200 + (
                 49 * np.cos(joints[1] + joints[2]) * np.cos(joints[0])) / 125],
        [np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[4]) * np.sin(joints[0]) - np.cos(joints[0]) * np.sin(
            joints[4]),
         - np.cos(joints[0]) * np.cos(joints[4]) - np.cos(joints[1] + joints[2] + joints[3]) * np.sin(
             joints[0]) * np.sin(joints[4]),
         -np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]),
         (109 * np.cos(joints[0])) / 1000 + (17 * np.cos(joints[1]) * np.sin(joints[0])) / 40 - (
                 19 * np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[0])) / 200 + (
                 49 * np.cos(joints[1] + joints[2]) * np.sin(joints[0])) / 125],
        [-np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[4]),
         np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4]),
         -np.cos(joints[1] + joints[2] + joints[3]),
         89 / 1000 - (49 * np.sin(joints[1] + joints[2])) / 125 - (17 * np.sin(joints[1])) / 40 - (
                 19 * np.cos(joints[1] + joints[2] + joints[3])) / 200],
        [0, 0, 0, 1]])


def tf0to6(joints):
    return np.array([
        [np.cos(joints[5]) * (
                np.sin(joints[0]) * np.sin(joints[4]) + np.cos(joints[1] + joints[2] + joints[3]) * np.cos(
            joints[0]) * np.cos(joints[4])) - np.sin(
            joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.sin(joints[5]), - np.sin(joints[5]) * (
                 np.sin(joints[0]) * np.sin(joints[4]) + np.cos(joints[1] + joints[2] + joints[3]) * np.cos(
             joints[0]) * np.cos(joints[4])) - np.sin(
            joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.cos(joints[5]),
         np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.sin(joints[4]) - np.cos(joints[4]) * np.sin(
             joints[0]),
         (17 * np.cos(joints[0]) * np.cos(joints[1])) / 40 - (109 * np.sin(joints[0])) / 1000 - (
                 np.cos(joints[4]) * np.sin(joints[0])) / 4 - (
                 19 * np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[0])) / 200 + (
                 49 * np.cos(joints[1] + joints[2]) * np.cos(joints[0])) / 125 + (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.sin(joints[4])) / 4],
        [- np.cos(joints[5]) * (
                np.cos(joints[0]) * np.sin(joints[4]) - np.cos(joints[1] + joints[2] + joints[3]) * np.cos(
            joints[4]) * np.sin(joints[0])) - np.sin(
            joints[1] + joints[2] + joints[3]) * np.sin(joints[0]) * np.sin(joints[5]), np.sin(joints[5]) * (
                 np.cos(joints[0]) * np.sin(joints[4]) - np.cos(joints[1] + joints[2] + joints[3]) * np.cos(
             joints[4]) * np.sin(joints[0])) - np.sin(
            joints[1] + joints[2] + joints[3]) * np.cos(joints[5]) * np.sin(joints[0]),
         np.cos(joints[0]) * np.cos(joints[4]) + np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]) * np.sin(
             joints[4]),
         (109 * np.cos(joints[0])) / 1000 + (np.cos(joints[0]) * np.cos(joints[4])) / 4 + (
                 17 * np.cos(joints[1]) * np.sin(joints[0])) / 40 - (
                 19 * np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[0])) / 200 + (
                 49 * np.cos(joints[1] + joints[2]) * np.sin(joints[0])) / 125 + (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]) * np.sin(joints[4])) / 4],
        [- np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[5]) - np.sin(
            joints[1] + joints[2] + joints[3]) * np.cos(joints[4]) * np.cos(joints[5]),
         np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[4]) * np.sin(joints[5]) - np.cos(
             joints[1] + joints[2] + joints[3]) * np.cos(joints[5]),
         -np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4]),
         89 / 1000 - (49 * np.sin(joints[1] + joints[2])) / 125 - (17 * np.sin(joints[1])) / 40 - (
                 np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4 - (
                 19 * np.cos(joints[1] + joints[2] + joints[3])) / 200],
        [0, 0, 0, 1]])


def position_jacobian(joints):
    return np.array([
        [(19 * np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[0])) / 200 - (
                    np.cos(joints[0]) * np.cos(joints[4])) / 4 - (
                 17 * np.cos(joints[1]) * np.sin(joints[0])) / 40 - (109 * np.cos(joints[0])) / 1000 - (
                 49 * np.cos(joints[1] + joints[2]) * np.sin(joints[0])) / 125 - (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[0]) * np.sin(joints[4])) / 4,
         -np.cos(joints[0]) * (
                 (19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (49 * np.sin(joints[1] + joints[2])) / 125 + (
                 17 * np.sin(joints[1])) / 40 + (np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4),
         -np.cos(joints[0]) * ((19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (
                     49 * np.sin(joints[1] + joints[2])) / 125 + (
                                       np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4),
         -np.cos(joints[0]) * (
                 (19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (
                 np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4),
         (np.sin(joints[0]) * np.sin(joints[4])) / 4 + (
                     np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.cos(joints[4])) / 4, 0],
        [(17 * np.cos(joints[0]) * np.cos(joints[1])) / 40 - (109 * np.sin(joints[0])) / 1000 - (
                    np.cos(joints[4]) * np.sin(joints[0])) / 4 - (
                 19 * np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[0])) / 200 + (
                 49 * np.cos(joints[1] + joints[2]) * np.cos(joints[0])) / 125 + (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[0]) * np.sin(joints[4])) / 4,
         -np.sin(joints[0]) * (
                 (19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (49 * np.sin(joints[1] + joints[2])) / 125 + (
                 17 * np.sin(joints[1])) / 40 + (np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4),
         -np.sin(
             joints[0]) * ((19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (
                     49 * np.sin(joints[1] + joints[2])) / 125 + (
                                   np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4),
         -np.sin(joints[0]) * (
                 (19 * np.cos(joints[1] + joints[2] + joints[3])) / 200 + (
                 np.sin(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4), (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.cos(joints[4]) * np.sin(joints[0])) / 4 - (
                 np.cos(joints[0]) * np.sin(joints[4])) / 4, 0],
        [0, np.sin(joints[1] + joints[2] + joints[3] - joints[4]) / 8 + (
                    19 * np.sin(joints[1] + joints[2] + joints[3])) / 200 - np.sin(
            joints[1] + joints[2] + joints[3] + joints[4]) / 8 - (49 * np.cos(joints[1] + joints[2])) / 125 - (
                 17 * np.cos(joints[1])) / 40, np.sin(joints[1] + joints[2] + joints[3] - joints[4]) / 8 + (
                 19 * np.sin(joints[1] + joints[2] + joints[3])) / 200 - np.sin(
            joints[1] + joints[2] + joints[3] + joints[4]) / 8 - (49 * np.cos(joints[1] + joints[2])) / 125, (
                 19 * np.sin(joints[1] + joints[2] + joints[3])) / 200 - (
                 np.cos(joints[1] + joints[2] + joints[3]) * np.sin(joints[4])) / 4, -(
                np.sin(joints[1] + joints[2] + joints[3]) * np.cos(joints[4])) / 4, 0]
    ])
