import argparse
import numpy as np
import faiss

from pathlib import Path
from tqdm import tqdm

DIR = 'concatenated'
feature_dir_prefix = f'/media/moon/T7 Shield/{DIR}'
image_dir_prefix = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/'
method_list = ['convap', 'cosplace', 'gem', 'mixvpr', 'netvlad', 'transvpr']
dir = 'concat'


def imagename_generator(image_dir_prefix, image_dir_postfix):
    image_dir = image_dir_prefix + image_dir_postfix + f'/{dir}'
    imagename_list = sorted(list(Path(image_dir).glob('*.png')))
    return [str(i)[len(image_dir_prefix):] for i in imagename_list]


def main():

    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--version', type=str, default='')

    options = args.parse_args()
    version = options.version

    # 0519: index
    # 0828: query
    index_imagename_list = imagename_generator(image_dir_prefix, '0519')
    query_imagename_list = imagename_generator(image_dir_prefix, '0828')

    # convap, cosplace, gem, mixvpr, netvlad, transvlad
    npy_files = sorted(list(Path(feature_dir_prefix).glob(f'*{version}.npy')))

    for i in tqdm(range(int(len(npy_files)/2))):

        save_list = []
        
        # 0519, 0828, ... 
        index = np.load(npy_files[2 * i]).astype('float32')[:-100]
        query = np.load(npy_files[2 * i + 1]).astype('float32')[:-100]

        pool_size = query.shape[1]

        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(index)

        _, predictions = faiss_index.search(query, min(len(index), 1))

        for idx, val in enumerate(predictions):
            # val: array([15916])
            save_list.append(f'{query_imagename_list[idx]} {index_imagename_list[val[0]]}')

        save_file_name = f'./eval/{DIR}/{method_list[i]}{version}_result.txt'

        with open(save_file_name, 'w') as file:
            file.write('# query index\n')

        with open(save_file_name, 'a') as file:
            for line in save_list:
                file.write(line + '\n')

def main2():

    # for test just one method


    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--version', type=str, default='1')
    args.add_argument('--save_dir', type=str, default='')

    options = args.parse_args()

    method = options.method + options.version

    # 0519: index
    # 0828: query
    index_imagename_list = imagename_generator(image_dir_prefix, '0519')
    query_imagename_list = imagename_generator(image_dir_prefix, '0828')

    # convap, cosplace, gem, mixvpr, netvlad, transvlad
    npy_files = sorted(list(Path(feature_dir_prefix).glob(f'{method}*.npy')))

    save_list = []
    
    index = np.load(npy_files[0]).astype('float32')[:-100]
    query = np.load(npy_files[1]).astype('float32')[:-100]

    pool_size = query.shape[1]

    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(index)

    _, predictions = faiss_index.search(query, min(len(index), 1))

    for idx, val in enumerate(predictions):
        # val: array([15916])
        save_list.append(f'{query_imagename_list[idx]} {index_imagename_list[val[0]]}')

    save_file_name = f'./eval/{options.save_dir}/{method}_result.txt'

    with open(save_file_name, 'w') as file:
        file.write('# query index\n')

    with open(save_file_name, 'a') as file:
        for line in save_list:
            file.write(line + '\n')

if __name__ == '__main__':
    # main()
    main2()