import numpy as np
import faiss

from pathlib import Path
from tqdm import tqdm

feature_dir_prefix = '/media/moon/T7 Shield/master_research'
image_dir_prefix = '/media/moon/moon_ssd/moon_ubuntu/icrca/'
method_list = ['convap', 'cosplace', 'gem', 'mixvpr', 'netvlad', 'transvlad']


def imagename_generator(image_dir_prefix, image_dir_postfix):
    image_dir = image_dir_prefix + image_dir_postfix
    imagename_list = sorted(list(Path(image_dir).glob('**/*.png')))
    return [str(i)[len(image_dir_prefix):] for i in imagename_list]


def main():
    # 0519: index
    # 0828: query
    index_imagename_list = imagename_generator(image_dir_prefix, '0519')
    query_imagename_list = imagename_generator(image_dir_prefix, '0828')

    # convap, cosplace, gem, mixvpr, netvlad, transvlad
    npy_files = sorted(list(Path(feature_dir_prefix).glob('*.npy')))

    for i in tqdm(range(int(len(npy_files)/2))):

        save_list = []
        
        index = np.load(npy_files[2 * i]).astype('float32')
        query = np.load(npy_files[2 * i + 1]).astype('float32')

        pool_size = query.shape[1]

        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(index)

        _, predictions = faiss_index.search(query, min(len(index), 1))

        for idx, val in enumerate(predictions):
            # val: array([15916])
            save_list.append(f'{query_imagename_list[idx]} {index_imagename_list[val[0]]}')

        save_file_name = f'./eval/vpr_results/{method_list[i]}_result.txt'

        with open(save_file_name, 'w') as file:
            file.write('# query index\n')

        with open(save_file_name, 'a') as file:
            for line in save_list:
                file.write(line + '\n')

def main2():

    method = 'transvlad8'

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

    save_file_name = f'./eval/vpr_results/{method}_result.txt'

    with open(save_file_name, 'w') as file:
        file.write('# query index\n')

    with open(save_file_name, 'a') as file:
        for line in save_list:
            file.write(line + '\n')

if __name__ == '__main__':
    main2()