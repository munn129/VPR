import argparse

from os.path import isfile, join, exists

from pathlib import Path
from math import pi, sin, cos, sqrt, atan2

def gps_to_error(lat1, lon1, lat2, lon2):
    # 지구의 넓이 반지름
    R = 6371.0072 # radius of the earth in KM
    lat_to_deg = lat2 * pi/180 - lat1 * pi/180
    long_to_deg = lon2 * pi/180 - lon1 * pi/180

    a = sin(lat_to_deg/2)**2 + cos(lat1 * pi/180) * cos(lat2 * pi/180) * sin(long_to_deg/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c

    return d * 1000 #meter


class GeoTagImage:
    def __init__(self, image_name: str, latitude: float, longitude: float, heading: float) -> None:
        self.image_name = image_name
        self.latitude = latitude
        self.longitude = longitude
        self.heading = heading
    
    def get_image_name(self) -> str:
        return self.image_name
    
    def get_latitude(self) -> float:
        return self.latitude
    
    def get_longitude(self) -> float:
        return self.longitude
    
    def get_heading(self) -> float:
        return self.heading


class GPS:
    def __init__(self, gps_path: str) -> None:
        if not isfile(gps_path):
            raise FileNotFoundError(f'{gps_path} is not exist.')
        
        self.gps_path = gps_path
        self.geo_tag_image_list = []
        self._read_gps()

    def _read_gps(self) -> None:
        with open(self.gps_path, 'r') as file:
            for line in file:
                line = line.split(' ')
                image_name = line[0].split('/')[-1]
                latitude = float(line[1])
                longitude = float(line[2])
                heading = float(line[3])
                self.geo_tag_image_list.append(GeoTagImage(image_name, latitude, longitude, heading))
    
    def get_geo_tag_image_list(self) -> list:
        return self.geo_tag_image_list
    

class Result:
    def __init__(self, result_path: str) -> None:

        if not isfile(result_path):
            raise FileNotFoundError(f'{result_path} is not exists.')
        
        self.result_path = result_path
        
        self.query_list = []
        self.retrieved_list = []
        self.retrieval_num = 0
        self._read_result()

    def _read_result(self) -> None:
        with open(self.result_path, 'r') as file:
            for line in file:
                if line[0] == '#': continue
                
                line = line.split('\n')[0]
                line = line.split(' ')
                self.query_list.append(line[0][1:].split('/')[-1])
                self.retrieved_list.append(line[1][1:].split('/')[-1])

        tmp_query_list = []
        
        for i in self.query_list:
            if i not in tmp_query_list:
                tmp_query_list.append(i)

        self.retrieval_num = int(len(self.query_list) / len(tmp_query_list))
        self.query_list = tmp_query_list[:]
    
    def get_query_list(self) -> list:
        return self.query_list
    
    def get_retrieved_list(self) -> list:
        return self.retrieved_list
    
    def get_retrieval_num(self) -> int:
        return self.retrieval_num


class Evaluation:
    def __init__(self, result: Result, query: GPS, db: GPS, is_save = False) -> None:
        self.result = result
        self.query = query
        self.db = db
        self.is_save = is_save
        
        self._db_list = []
        self._db_filtering()

        self.eval_q_list = [i for i in self.query.get_geo_tag_image_list() if i.get_image_name() in self.result.get_query_list()]
        self.eval_r_list = []
        for i in self._db_list:
            for j in self.db.get_geo_tag_image_list():
                if i == j.get_image_name():
                    self.eval_r_list.append(j)

        self.translation_error = []
        self.rotation_error = []

    def _db_filtering(self) -> None:
        for i in range(len(self.result.get_query_list())):
           self._db_list.append(self.result.get_retrieved_list()[i * self.result.get_retrieval_num()])

    def error_calculator(self) -> None:
        for i in range(len(self.eval_q_list)):
            geotag_q = self.eval_q_list[i]
            geotag_r = self.eval_r_list[i]
            self.translation_error.append(gps_to_error(geotag_q.get_latitude(),
                                                       geotag_q.get_longitude(),
                                                       geotag_r.get_latitude(),
                                                       geotag_r.get_longitude()))
            self.rotation_error.append(abs(abs(geotag_q.get_heading()) - abs(geotag_r.get_heading())))

    def save(self, dir) -> None:
        with open(dir, 'w') as file:
            file.write('# tranlation rotation\n')
        
        with open(dir, 'a') as file:
            for i, j in zip(self.translation_error, self.rotation_error):
                file.write(f'{i} {j}\n')

    def error_analysis(self) -> None:
        '''
        TODO
        translation error, rotation error
        + by range -> have to find critia
        recall rate: translation, rotation, both
        Pearson correlation coefficient -> environment diagram
        '''
        print(f'Average translation error: {sum(self.translation_error)/len(self.translation_error)}')
        print(f'Average rotation error: {sum(self.rotation_error)/len(self.rotation_error)}')


method_list = ['convap', 'cosplace', 'gem', 'mixvpr', 'netvlad', 'transvpr']

dir = 'concat'
query_gps_dir = f'/home/moon/Documents/VPR/eval/0828_{dir}_gt.txt'
index_gps_dir = f'/home/moon/Documents/VPR/eval/0519_{dir}_gt.txt'


def main():

    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--version', type=str, default='1')

    options = args.parse_args()
    version = options.version

    result_dir = Path('/home/moon/Documents/VPR/eval/multiview_results')
    save_dir = '/home/moon/Documents/VPR/eval/multiview_error'

    result_files = sorted(list(result_dir.glob(f'*{version}_result.txt')))

    index_gps = GPS(index_gps_dir)
    query_gps = GPS(query_gps_dir)

    for result_file, method in zip(result_files, method_list):

        result = Result(result_file)
        eval = Evaluation(result, query_gps, index_gps)
        eval.error_calculator()
        eval.save(f'{save_dir}/{method}{version}.txt')
        eval.error_analysis()


def main2():

    # for test just one method

    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--version', type=str, default='1')

    options = args.parse_args()

    method = options.method + options.version

    result_dir = f'/home/moon/Documents/VPR/eval/multiview_results/{method}_result.txt'
    save_dir = '/home/moon/Documents/VPR/eval/error_results'

    index_gps = GPS(index_gps_dir)
    query_gps = GPS(query_gps_dir)

    result = Result(result_dir)
    eval = Evaluation(result, query_gps, index_gps)
    eval.error_calculator()
    eval.save(f'{save_dir}/{method}.txt')
    eval.error_analysis()


if __name__ == '__main__':
    # main()
    main2()


# translation error
# 'convap': 554
# 'cosplace': 17
# 'gem': 374
# 'mixvpr': 11
# 'netvlad': 136
# 'transvlad': 516
# 'transvpr': 122

# after 2 is subset
# transvlad2: 290
# netvlad2: 470
# transvlad3: 281
# mixvpr2: 429
# transvlad4: 299 -> only attention map
# transvlad5: 516
# transvlad6: 516
# transvlad7: 549
# transvlad8: 516 -> only patch
# transvlad9: 512 -> 0 or 1 20%
# 11: 467 -> attention map only
# 12: 488 -> attention map * image -> mixvpr
# 13: only mixvpr
# 14: 340 -> 12 with normalize

# multiview
# convap: 554 -> 552 
# cosplace: 17 -> 66
# gem: 374 -> 336
# mixvpr: 11 -> 14
# netvlad: 136 -> 110
# transvpr: 122 -> 64

# 1: only front
# convap: 630
# cosplace: 22
# gem: 404
# mixvpr: 21
# netvlad: 145
# transvpr: 127

# transvlad1: probmap + cosplace: something2: 55
# transvlad2: sum: something3: 52.8966
# transvlad3: mixvpr mean attention: soemthing4: 52.84
# transvlad4: mixvpr with channel projection + mean: st4: 52.8967
# transvlad5: mixvpr with ch proj, sum, st4: 52.8967
# transvlad6: mixvpr --mix--> cosplace(only agg): st5: 35.427
# transvlad7: mixvpr, after projection: st5: 56.477
# transvlad8: 6 with only front: 53.67
# scenario 9: st5 + transvpr: 56.581
# 10: st6 + inter(mix + trans): 55.98
# 11: mix -> cos, trans -> cos, +, st7: 47.939
# 12: 11 with normalization: 51.007
# 13: 4 image each mix and GeM/ something8 (image size 640640): 23.71 
# 14: ablation: without mix: 42.404
# 15: 320*320 with scenario 13: 28.38

# need to image re-processing