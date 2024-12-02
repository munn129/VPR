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

DIR = 'concatenated'

def main():

    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, default='transvlad',
                      help='VPR method name, e.g., netvlad, cosplace, mixpvr, gem, convap, transvpr')
    args.add_argument('--version', type=str, default='1')

    options = args.parse_args()
    version = options.version

    result_dir = Path(f'/home/moon/Documents/VPR/eval/{DIR}')
    save_dir = f'/home/moon/Documents/VPR/eval/{DIR}_analysis'

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
    args.add_argument('--matching_dir', type=str, default='')
    args.add_argument('--save_dir', type=str, default='')

    options = args.parse_args()

    method = options.method + options.version

    result_dir = f'/home/moon/Documents/VPR/eval/concatenated_result/{method}_result.txt'
    save_dir = '/home/moon/Documents/VPR/eval/concatenated'

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

# need to image re-processing -> cancel, because front image size is different from other images

# image size 1024 * 1024/concat
# convap: 560.798
# cosplace: 66.439
# gem: 336.820
# mixvpr: 14.336
# netvlad:  111.336
# transvpr: 65.028

# image size 1024 / front
# convap: 607
# cosplace: 23.27
# gem: 403.88
# mixvpr: 21.76
# netvlad: 145.47
# transvpr: 128.127

# image size 640 640 / concat
# convap: 601.29
# cosplace: 66.81
# gem: 340.21
# mixvpr: 14.53
# netvlad: 114.12
# transvpr: 65.09

# image size 640 / front
# convap: 600.498
# cosplace: 23.529
# gem: 403.254
# mixvpr: 21.869
# netvlad: 145.670
# transvpr: 127.653

# scenario 16, something9, twist, concat, mean: 25.61
# 17/something9/twist/front/mean: 61.467
# sc17, smth9, twt, cc, sum: 25.61

# vertical cocat
# convap: 689
# cosplace: 87
# gem: 379
# mixvpr: 73
# netvlad: 253
# transvpr: 211

# horizontal concat
# convap: 566
# cosplace: 138.847
# gem: 412.346
# mixvpr: 29.407
# netvlad: 207.185
# transvpr: 161.267

# dim ex
# 1280_concat_4096 netvlad: 67.145
# 1280_concat_512 netvlad: 110.604
# 1280_concat_128 netvlad: 194.918
# 640_concat_4096 netvlad: 69.144
# 640_concat_512 netvald: 113.978
# 640_concat_128 netvlad: 194.036
# 320_concat_4096 netvlad: 66.808
# 320_concat_512 netvlad: 110.081
# 320_concat_128 netvlad: 194.965

# 1280_front_4096 netvlad: 119.517
# 1280_front_512 netvlad: 145.344
# 1280_front_128 netvlad: 205.947
# 640_front_4096 netvlad: 121.001
# 640_front_512 netvlad: 145.550
# 640_front_128 netvlad: 206.349
# 320_front_4096 netvlad: 120.032
# 320_front_512 netvlad: 145.501
# 320_front_128 netvlad: 206.585

# 1280_concat_2048 convap 608.253
# 1280_concat_1024 convap 621.812
# 1280_concat_512 convap 608.404
# 1280_concat_256 convap 667.295
# 1280_concat_128 convap 694.707
# 640_concat_2048 convap 668.451
# 640_concat_1024 convap 568.445
# 640_concat_512 convap 631.219
# 640_concat_256 convap 564.449
# 640_concat_128 convap 549.611
# 320_concat_2048 convap 571.859
# 320_concat_1024 convap 593.428
# 320_concat_512 convap 729.207
# 320_concat_256 convap 604.873
# 320_concat_128 convap 613.410

# 1280_front_2048 convap 565.405
# 1280_front_1024 convap 587.518
# 1280_front_512 convap 621.713
# 1280_front_256 convap 519.370
# 1280_front_128 convap 547.198
# 640_front_2048 convap 604.736
# 640_front_1024 convap 643.931
# 640_front_512 convap 506.473
# 640_front_256 convap 653.620
# 640_front_128 convap 673.632
# 320_front_2048 convap 571.859
# 320_front_1024 convap 593.428
# 320_front_512 convap 729.207
# 320_front_256 convap 604.873
# 320_front_128 convap
#TODO need to 4096 dim

# 1280_concat_2048 cosplace 57.013
# 1280_concat_1024 cosplace 64.305
# 1280_concat_512 cosplace 66.648
# 1280_concat_256 cosplace 58.870
# 1280_concat_128 cosplace 64.421
# 640_concat_2048 cosplace 57.312
# 640_concat_1024 cosplace 64.288
# 640_concat_512 cosplace 66.689
# 640_concat_256 cosplace 58.276
# 640_concat_128 cosplace 126.656
# 320_concat_2048 cosplace 56.731
# 320_concat_1024 cosplace 64.643
# 320_concat_512 cosplace 66.302
# 320_concat_256 cosplace 58.914
# 320_concat_128 cosplace 64.925

# 1280_front_2048 cosplace 33.566
# 1280_front_1024 cosplace 25.127
# 1280_front_512 cosplace 23.054
# 1280_front_256 cosplace 24.301
# 1280_front_128 cosplace 66.216
# 640_front_2048 cosplace 33.327
# 640_front_1024 cosplace 24.424
# 640_front_512 cosplace 23.584
# 640_front_256 cosplace 24.920
# 640_front_128 cosplace 61.132
# 320_front_2048 cosplace 33.362
# 320_front_1024 cosplace 24.762
# 320_front_512 cosplace 22.926
# 320_front_256 cosplace 24.459
# 320_front_128 cosplace 65.555

# 1280_concat_2048 mmt 20.296
# 1280_concat_1024 mmt 21.450
# 1280_concat_512 mmt 25.500
# 1280_concat_256 mmt 33.633
# 1280_concat_128 mmt 71.273
# 640_concat_2048 mmt 20.532
# 640_concat_1024 mmt 21.534
# 640_concat_512 mmt 25.344
# 640_concat_256 mmt 33.377
# 640_concat_128 mmt 71.510
# 320_concat_2048 mmt 20.423
# 320_concat_1024 mmt 21.427
# 320_concat_512 mmt 25.614
# 320_concat_256 mmt 33.387
# 320_concat_128 mmt 71.893

# 1280_front_2048 mmt 66.667
# 1280_front_1024 mmt 63.642
# 1280_front_512 mmt 61.563
# 1280_front_256 mmt 81.211
# 1280_front_128 mmt 110.627
# 640_front_2048 mmt 66.977
# 640_front_1024 mmt 63.841
# 640_front_512 mmt 61.573
# 640_front_256 mmt 82.172
# 640_front_128 mmt 109.613
# 320_front_2048 mmt 66.718
# 320_front_1024 mmt 63.957
# 320_front_512 mmt 61.467
# 320_front_256 mmt 81.380
# 320_front_128 mmt 111.233