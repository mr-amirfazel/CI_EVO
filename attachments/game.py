import copy
import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt

MUTATION_PROB = 0

class Game:
    def __init__(self, levels):
        # Get a list of strings as levels
        # Store level length to determine if a sequence of action passes all the steps

        self.levels = levels
        self.current_level_index = -1
        self.current_level_len = 0

    def load_next_level(self):
        self.current_level_index += 1
        self.current_level_len = len(self.levels[self.current_level_index])

    def get_score(self, actions, eval_flag):
        # Get an action sequence and determine the steps taken/score
        # Return a tuple, the first one indicates if these actions result in victory
        # and the second one shows the steps taken
            current_level = self.levels[self.current_level_index]
            max_score = 0
            current_score = 0
            flag_end = True
            score = []

            for i in range(self.current_level_len):
                current_step = current_level[i]

                if current_step == '_' and actions[i] == '1' and i == self.current_level_len - 1:
                    current_score += 1
                if current_step == '_':
                    current_score += 1
                elif current_step == 'G' and current_level[i - 1] != 'L' and i >= 1 and actions[i - 2] == '1':
                    current_score += 2
                elif current_step == 'G' and actions[i - 1] == '1':
                    current_score += 1
                elif current_step == 'L' and actions[i - 1] == '2':
                    current_score += 1
                elif current_step == 'M' and actions[i - 1] == '1':
                    current_score += 1
                elif current_step == 'M' and actions[i - 1] != '1':
                    current_score += 2
                elif i == self.current_level_len - 1:
                    break
                else:
                    score.append(current_score)
                    flag_end = False
                    current_score = 0

            score.append(current_score)

            if eval_flag and flag_end:
                return max(score) + 5, True
            elif eval_flag and not flag_end:
                return max(score), False
            elif flag_end and not eval_flag:
                return max(score) + 5
            else:
                return max(score)


def get_data(name):
    file = open(name, 'r')
    str = file.readline()
    return len(str), str


def population(size, chromosome_count):
    chromosomes = \
        [[np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1]) for i in range(size)] for j in range(chromosome_count)]
    return chromosomes


def get_chromosome(chromosome, count, length):
    chromosomes = []
    for i in range(count):
        s = ""
        for j in range(length):
            s += str(chromosome[i][j])
        chromosomes.append(s)
    return chromosomes


def fitness(chromosome, actions, chromosome_count, eval_flag):
    game = Game([actions])
    game.load_next_level()
    scores = []
    res = {}
    for i in range(chromosome_count):
        # game.set_eval_flag(eval_flag)
        if not eval_flag:
            scores.append(game.get_score(chromosome[i], eval_flag))
        else:
            score, end_flag = game.get_score(chromosome[i], eval_flag)
            res.setdefault(i, []).append(score)
            res.setdefault(i, []).append(end_flag)
            res.setdefault(i, []).append(chromosome[i])

    if eval_flag:
        return res
    else:
        return scores


def selection(scores, chromosome):
    scores_copy = scores
    scores_copy = np.sort(scores_copy)
    select_score = []
    select_chromosome_score = {}
    selected_number = []
    for i in range(int(len(scores) / 2)):
        for j in range(len(scores)):
            if scores[j] == scores_copy[-i - 1] and j not in selected_number:
                selected_number.append(j)
                select_score.append(scores[j])
                select_chromosome_score.setdefault(i, []).append(scores[j])
                select_chromosome_score.setdefault(i, []).append(chromosome[j])
                break
    return select_score, select_chromosome_score


def crossover(scores, chromosome, count):
    new_chromosome = []
    for i in range(int(count / 10)):
        new_chromosome.append(chromosome[i][1])
    for _ in range(len(scores) - int(count / 20)):
        p1, p2 = random.sample(range(100), 2)

        parent1 = chromosome[int(p1)][1]
        parent2 = chromosome[int(p2)][1]

        pt = random.randint(1, len(parent1) - 2)
        pt2 = random.randint(1, len(parent2) - 2)

        c1 = parent1[:pt] + parent2[pt:pt2] + parent1[pt2:]
        c2 = parent2[:pt] + parent1[pt:pt2] + parent2[pt2:]

        new_chromosome.append(c1)
        new_chromosome.append(c2)

    return new_chromosome


def mutation(chromosome):
    new_chromosome = chromosome.copy()
    for i in range(int(MUTATION_PROB * len(chromosome))):
        if i not in new_chromosome:
            new_chromosome.append(i)
            pt = random.randint(1, len(chromosome[i]) - 1)

            if chromosome[i][pt] == "1":
                new_chromosome[i] = chromosome[i][:pt] + "0" + chromosome[i][pt + 1:]
            else:
                x = np.random.choice([0, 2], p=[0.95, 0.05])
                new_chromosome[i] = chromosome[i][:pt] + str(x) + chromosome[i][pt + 1:]

    return new_chromosome

def evaluation(chromosome, actions, chromosome_count):
    final = fitness(chromosome, actions, chromosome_count, True)
    sorted_chromosomes = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return sorted_chromosomes


import matplotlib.pyplot as plt


def genetic(file, chromosome_count, iteration_count):
    length, moves = get_data(file)
    chromosomes = population(length, chromosome_count)
    chromosomes_str = get_chromosome(chromosomes, chromosome_count, length)
    scores_level = []
    max_scores = []
    min_scores = []

    for i in range(iteration_count):
        if i == 0:
            scores = fitness(chromosomes_str, moves, chromosome_count, False)
        else:
            scores = fitness(chromosomes_str, moves, chromosome_count, False)

        sorted_scores = sorted(scores)
        max_scores.append(sorted_scores[-1])
        min_scores.append(sorted_scores[0])
        scores_level.append(sum(scores) / len(scores))

        if i > 0 and abs(scores_level[i] - scores_level[i - 1]) < 0.0000000000001:
            print(f"Convergence has occurred in the {i} generation")
            break

        selected_score, selected_chromosomes = selection(scores, chromosomes_str)
        new_chromosomes = crossover(selected_score, selected_chromosomes, chromosome_count)
        chromosomes_str = mutation(new_chromosomes)

    sort_answer = evaluation(chromosomes_str, moves, chromosome_count)

    plt.plot(max_scores, label="max")
    plt.plot(min_scores, label="min")
    plt.plot(scores_level, label="avg")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend()
    plt.show()

    for i in range(len(sort_answer)):
        if sort_answer[i][1][1]:
            return sort_answer[i][1][0], sort_answer[i][1][1], sort_answer[i][1][2], moves

    return sort_answer[0][1][0], sort_answer[0][1][1], sort_answer[0][1][2], moves


def show_path(chromoseme, moves):
    for i in range(len(moves) + 1):
        # print(moves)
        print(chromoseme)
        for j in range(len(moves)):
            if j > 2 and chromoseme[i - 1] == "1" and i == j and chromoseme[i - 2] == "1" and chromoseme[i - 3] == "1":
                print("A ", end='')
            elif j > 1 and chromoseme[i - 1] == "1" and i == j and chromoseme[i - 2] != "1":

                print("A ", end='')
            elif j == 1 and i == j and chromoseme[i - 1] == "1":
                print("A ", end='')

            elif moves[j] == 'L':
                print("L ", end='')
            else:
                print("  ", end='')

        print()
        for j in range(len(moves)):
            if i == 0 and j == 0:
                print("A ", end='')

            elif j > 0 and chromoseme[i - 1] == "2" and i == j:

                print("a ", end='')

            elif j > 0 and j == i and chromosomes[i - 1] == "0":
                print("A ", end='')
            elif moves[j] == 'G' and chromosomes[j - 2] == "1" and i >= 2 and j < i:
                print("_ ", end='')

            elif moves[j] == 'G':
                print("G ", end='')
            # elif moves[j] == 'M' and i > 0 and chromoseme[i - 1] != "1" and i == j:
            #     print("A ", end='')
            elif moves[j] == 'M' and chromosomes[j - 2] != "1" and i >= 2 and j < i:
                print("_ ", end='')
            elif moves[j] == 'M':
                print("M ", end='')
            else:
                print("_ ", end='')

        time.sleep(0.2)

    return 0

if __name__ == "__main__":
    file_names = "./levels/level8.txt"
    chromosomes_count = 500
    iteration_count = 300
    MUTATION_PROB = 0.5
    score, end, chromosomes, action = genetic(file_names, chromosomes_count, iteration_count)
    if end:
        print("pass the level with " + str(score) + " score and  with " + chromosomes + " moves.")
        show_path(chromosomes, action)
        time.sleep(0.2)

    else:
        print("cant pass the level best score : " + str(score) + "  with " + chromosomes + " moves.")