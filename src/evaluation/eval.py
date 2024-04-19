import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def write_test_eval(test_loader, model, device, log_file):
    log_file = open(log_file, 'a')
    #for true positives
    true_stars = 0
    true_galaxies = 0
    true_qso = 0

    #for false positives
    false_stars = 0
    false_galaxies = 0
    false_qso = 0

    #for missed classifications
    missed_stars = 0
    missed_galaxies = 0
    missed_qso = 0

    false_star_galaxy = 0
    false_star_qso = 0
    false_galaxy_star = 0
    false_galaxy_qso = 0
    false_qso_star = 0
    false_qso_galaxy = 0

    total_stars = 0
    total_galaxies = 0
    total_qso = 0

    #test model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            print(probabilities)

            avg_galaxy_confidence = torch.mean(probabilities[:, 0]).item()
            avg_qso_confidence = torch.mean(probabilities[:, 1]).item()
            avg_star_confidence = torch.mean(probabilities[:, 2]).item()

            total += labels.size(0)
            total_stars += labels[labels == 0].size(0)
            total_galaxies += labels[labels == 1].size(0)
            total_qso += labels[labels == 2].size(0)
            correct += (predicted == labels).sum().item()

            true_stars += ((predicted == 0) & (labels == 0)).sum().item()
            true_galaxies += ((predicted == 1) & (labels == 1)).sum().item()
            true_qso += ((predicted == 2) & (labels == 2)).sum().item()

            false_stars += ((predicted == 0) & (labels != 0)).sum().item()
            false_galaxies += ((predicted == 1) & (labels != 1)).sum().item()
            false_qso += ((predicted == 2) & (labels != 2)).sum().item()

            missed_stars += ((predicted != 0) & (labels == 0)).sum().item()
            missed_galaxies += ((predicted != 1) & (labels == 1)).sum().item()
            missed_qso += ((predicted != 2) & (labels == 2)).sum().item()

            false_star_galaxy += ((predicted == 0) & (labels == 1)).sum().item()
            false_star_qso += ((predicted == 0) & (labels == 2)).sum().item()
            false_galaxy_star += ((predicted == 1) & (labels == 0)).sum().item()
            false_galaxy_qso += ((predicted == 1) & (labels == 2)).sum().item()
            false_qso_star += ((predicted == 2) & (labels == 0)).sum().item()
            false_qso_galaxy += ((predicted == 2) & (labels == 1)).sum().item()



    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    log_file.write(f'Accuracy on test set: {accuracy * 100:.2f}%\n')
    # class accuracy percentages
    # total_stars = true_stars + missed_stars
    # total_galaxies = true_galaxies + missed_galaxies
    # total_qso = true_qso + missed_qso
    true_stars = true_stars / total_stars
    false_stars = false_stars / total_stars
    missed_stars = missed_stars / total_stars

    true_galaxies = true_galaxies / total_galaxies
    false_galaxies = false_galaxies / total_galaxies
    missed_galaxies = missed_galaxies / total_galaxies

    true_qso = true_qso / total_qso
    false_qso = false_qso / total_qso
    missed_qso = missed_qso / total_qso

    #confusion matrix percentages
    false_star_galaxy = false_star_galaxy / total_galaxies
    false_star_qso = false_star_qso / total_qso
    false_galaxy_star = false_galaxy_star / total_stars
    false_galaxy_qso = false_galaxy_qso / total_qso
    false_qso_star = false_qso_star / total_stars
    false_qso_galaxy = false_qso_galaxy / total_galaxies

    star_precision = true_stars / (true_stars + false_stars)
    star_recall = true_stars / (true_stars + missed_stars)
    star_f1 = 2 * (star_precision * star_recall) / (star_precision + star_recall)

    galaxy_precision = true_galaxies / (true_galaxies + false_galaxies)
    galaxy_recall = true_galaxies / (true_galaxies + missed_galaxies)
    galaxy_f1 = 2 * (galaxy_precision * galaxy_recall) / (galaxy_precision + galaxy_recall)

    qso_precision = true_qso / (true_qso + false_qso)
    qso_recall = true_qso / (true_qso + missed_qso)
    qso_f1 = 2 * (qso_precision * qso_recall) / (qso_precision + qso_recall)


    # log_file.write(f'True stars: {true_stars * 100:.2f}%\n')
    # log_file.write(f'False stars: {false_stars * 100:.2f}%\n')
    # log_file.write(f'Missed stars: {missed_stars * 100:.2f}%\n\n')
    # log_file.write(f'True galaxies: {true_galaxies * 100:.2f}%\n')
    # log_file.write(f'False galaxies: {false_galaxies * 100:.2f}%\n')
    # log_file.write(f'Missed galaxies: {missed_galaxies * 100:.2f}%\n\n')
    # log_file.write(f'True qso: {true_qso * 100:.2f}%\n')
    # log_file.write(f'False qso: {false_qso * 100:.2f}%\n')
    # log_file.write(f'Missed qso: {missed_qso * 100:.2f}%\n')

    log_file.write(f'Star precision: {star_precision * 100:.2f}%\n')
    log_file.write(f'Star recall: {star_recall * 100:.2f}%\n')
    log_file.write(f'Star F1 score: {star_f1 * 100:.2f}%\n\n')

    log_file.write(f'Galaxy precision: {galaxy_precision * 100:.2f}%\n')
    log_file.write(f'Galaxy recall: {galaxy_recall * 100:.2f}%\n')
    log_file.write(f'Galaxy F1 score: {galaxy_f1 * 100:.2f}%\n\n')

    log_file.write(f'QSO precision: {qso_precision * 100:.2f}%\n')
    log_file.write(f'QSO recall: {qso_recall * 100:.2f}%\n')
    log_file.write(f'QSO F1 score: {qso_f1 * 100:.2f}%\n\n')

    #create confusion matrix
    matrix = np.array([[true_stars, false_galaxies, false_qso],
                        [false_star_galaxy, true_galaxies, false_qso_galaxy],
                        [false_star_qso, false_galaxy_qso, true_qso]])
    log_file.write(f'Confusion matrix:\n{matrix}\n\n')
    # log_file.write(f'Stars classified as galaxies: {false_galaxy_star * 100:.2f}%\n')
    # log_file.write(f'Stars classified as QSO: {false_qso_star * 100:.2f}%\n')

    # log_file.write(f'Galaxies classified as stars: {false_star_galaxy * 100:.2f}%\n')
    # log_file.write(f'Galaxies classified as QSO: {false_qso_galaxy * 100:.2f}%\n')

    # log_file.write(f'QSO classified as stars: {false_star_qso * 100:.2f}%\n')
    # log_file.write(f'QSO classified as galaxies: {false_galaxy_qso * 100:.2f}%\n\n')


    log_file.write(f'Average star confidence: {avg_star_confidence * 100:.2f}%\n')
    log_file.write(f'Average galaxy confidence: {avg_galaxy_confidence * 100:.2f}%\n')
    log_file.write(f'Average qso confidence: {avg_qso_confidence * 100:.2f}%\n\n')

    log_file.close()
    return 1