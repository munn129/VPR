#!/usr/bin/env python3
from pathlib import Path
from os.path import isfile, join, exists
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

analysis_save_prefix = './concatenated_analysis'

# multiview sync data only front result
front_convap = './multiview_error/convap1.txt'
front_cosplace = './multiview_error/cosplace1.txt'
front_gem = './multiview_error/gem1.txt'
front_mixvpr = './multiview_error/mixvpr1.txt'
front_netvlad = './multiview_error/netvlad1.txt'
front_transvpr = './multiview_error/transvpr1.txt'
front_something = './error_results/transvlad8.txt'

# multiview image
multi_convap = './multiview_error/convap.txt'
multi_cosplace = './multiview_error/cosplace.txt'
multi_gem = './multiview_error/gem.txt'
multi_mixvpr = './multiview_error/mixvpr.txt'
multi_netvlad = './multiview_error/netvlad.txt'
multi_transvpr = './multiview_error/transvpr.txt'
multi_something = './error_results/transvlad6.txt'

file_list = [front_convap, front_cosplace, front_gem, front_mixvpr, front_netvlad, front_transvpr, front_something,
             multi_convap, multi_cosplace, multi_gem, multi_mixvpr, multi_netvlad, multi_transvpr, multi_something]

name_list = ['front_convap', 'front_cosplace', 'front_gem', 'front_mixvpr', 'front_netvlad', 'front_transvpr', 'front_something',
             'multi_convap', 'multi_cosplace', 'multi_gem', 'multi_mixvpr', 'multi_netvlad', 'multi_transvpr', 'multi_something']

def dictionary_updater(cnt_dict, critia) -> None:
    try:
        cnt_dict[critia] += 1
    except:
        cnt_dict[critia] = 1

def main():

    t_err_check_list = [1, 2.5, 5, 7.5, 10]
    r_err_check_list = [1, 2.5, 5, 7.5, 10]
    if len(t_err_check_list) != len(r_err_check_list):
        raise Exception('check lists are not same length')

    for result, name in zip(file_list, name_list):
        
        translation_error_list = []
        rotation_error_list = []

        cnt_dict = {}

        with open(result, 'r') as file:
            for line in file:
                if line[0] == '#': continue

                line = line.split('\n')[0]
                line = line.split(' ')

                translation_error = float(line[0])
                rotation_error = float(line[1])
                
                translation_error_list.append(translation_error)
                rotation_error_list.append(rotation_error)

                # recall(only translation error)
                for i in t_err_check_list:
                    if translation_error < float(i):
                        dictionary_updater(cnt_dict, f'{str(i)}_m')

                # recall(only rotation error)
                for i in r_err_check_list:
                    if rotation_error < float(i):
                        dictionary_updater(cnt_dict, f'{str(i)}_degree')

                # recall(translation and rotation error)
                for t, r in zip(t_err_check_list, r_err_check_list):
                    if translation_error < float(t) and rotation_error < float(r):
                        dictionary_updater(cnt_dict, f'{str(t)}_m_and_{str(r)}_degree')


        print(f'########## Result of {name} ##########')

        with open(join(analysis_save_prefix, f'{name}.txt'), 'w') as file:
            # recall rate
            for i in cnt_dict:
                sentence = f'recall rate @ {i}: {cnt_dict[i]/len(translation_error_list) *100} %'
                file.write(f'{sentence}\n')
                print(sentence)

            file.write('\n')

            # average translation error, min, max, median
            file.write(f'########## TRANSLATION ERROR ##########\n')
            file.write(f'average: {sum(translation_error_list)/len(translation_error_list)}\n')
            file.write(f'min: {min(translation_error_list)}\n')
            file.write(f'max: {max(translation_error_list)}\n')
            file.write(f'median: {translation_error_list[int(len(translation_error_list)/2)]}\n')

            # average rotation error
            file.write(f'########## ROTATION ERROR ##########\n')
            file.write(f'average: {sum(rotation_error_list)/len(rotation_error_list)}\n')
            file.write(f'min: {min(rotation_error_list)}\n')
            file.write(f'max: {max(rotation_error_list)}\n')
            file.write(f'median: {rotation_error_list[int(len(rotation_error_list)/2)]}\n')

            # trimmed translation error(critia: translation error)


        print(f'{name} is saved')


def main2():

    t_err_check_list = [1, 2.5, 5, 7.5, 10]
    r_err_check_list = [1, 2.5, 5, 7.5, 10]

    analysis_save_prefix = './late'
    result_dir = './late'
    result_files = sorted(list(Path(result_dir).glob('*.txt')))

    for result in tqdm(result_files):

        translation_error_list = []
        rotation_error_list = []
        cnt_dict = {}

        name = str(result).split('/')[-1][:-4]

        with open(result, 'r') as file:
            for line in file:
                if line[0] == '#': continue

                line = line.split('\n')[0]
                line = line.split(' ')

                translation_error = float(line[0])
                rotation_error = float(line[1])
                
                translation_error_list.append(translation_error)
                rotation_error_list.append(rotation_error)

                # recall(only translation error)
                for i in t_err_check_list:
                    if translation_error < float(i):
                        dictionary_updater(cnt_dict, f'{str(i)}_m')

                # recall(only rotation error)
                for i in r_err_check_list:
                    if rotation_error < float(i):
                        dictionary_updater(cnt_dict, f'{str(i)}_degree')

                # recall(translation and rotation error)
                for t, r in zip(t_err_check_list, r_err_check_list):
                    if translation_error < float(t) and rotation_error < float(r):
                        dictionary_updater(cnt_dict, f'{str(t)}_m_and_{str(r)}_degree')


        print(f'########## Result of {name} ##########')

        with open(join(analysis_save_prefix, f'{name}.txt'), 'w') as file:
            # recall rate
            for i in sorted(cnt_dict.keys()):
                sentence = f'recall rate @ {i}: {cnt_dict[i]/len(translation_error_list) *100} %'
                file.write(f'{sentence}\n')
                print(sentence)

            file.write('\n')

            # average translation error, min, max, median
            file.write(f'########## TRANSLATION ERROR ##########\n')
            file.write(f'average: {sum(translation_error_list)/len(translation_error_list)}\n')
            file.write(f'min: {min(translation_error_list)}\n')
            file.write(f'max: {max(translation_error_list)}\n')
            file.write(f'median: {translation_error_list[int(len(translation_error_list)/2)]}\n')

            # average rotation error
            file.write(f'########## ROTATION ERROR ##########\n')
            file.write(f'average: {sum(rotation_error_list)/len(rotation_error_list)}\n')
            file.write(f'min: {min(rotation_error_list)}\n')
            file.write(f'max: {max(rotation_error_list)}\n')
            file.write(f'median: {rotation_error_list[int(len(rotation_error_list)/2)]}\n')

            # trimmed translation error(critia: translation error)


        print(f'{name} is saved')

if __name__ == '__main__':
    main2()