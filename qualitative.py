from pathlib import Path
from PIL import Image
from tqdm import tqdm


'''
front
idx: 9100
convap: 750.9724167177377
cosplace: 1.4247304821580107
netvlad: 837.7636891080024
transvlad: 1.6784130832548971
gem: 1139.7200344653677
transvpr: 837.7636891080024
ablation: 1.4247304821580107

idx: 8850
convap: 1022.5649574865365
cosplace: 2.9800076997828517
netvlad: 579.4472350314091
transvlad: 1.3500452109571095
gem: 610.6609426187158
transvpr: 612.9498337002959
ablation: 4.659521133425168

idx: 6500
convap: 759.0020927598205
cosplace: 1.5282341939830113
netvlad: 851.2834368921712
transvlad: 0.946633778377577
gem: 792.6573955699339
transvpr: 787.0640790666425
ablation: 0.946633778377577

idx: 4394
convap: 769.193476409804
cosplace: 4.489399703911394
netvlad: 900.6145187756129
transvlad: 85.88012627155693
gem: 910.9889884783673
transvpr: 649.6833365766885
ablation: 4.561860989138762

idx: 323
convap: 902.9984021695636
cosplace: 1.1596271151381932
netvlad: 944.74635460475
transvlad: 0.7977478091951187
gem: 1033.1973894011448
transvpr: 953.8709051125389
ablation: 0.9600796373731029

idx: 4638
convap: 431.0827259977026
cosplace: 3.998565337048129
netvlad: 150.15936301591606
transvlad: 504.52110894437465
gem: 392.8003718997474
transvpr: 328.8196962062826
ablation: 504.22376631472633

idx: 8171
convap: 280.6953220247618
cosplace: 11.852490425639576
netvlad: 8.754717088933253
transvlad: 1114.499495773835
gem: 13.975852975727342
transvpr: 6.731116248795105
ablation: 947.2098749894906

concat
==========================================
idx: 393
convap: 800.1141820276108
cosplace: 850.0996085827943
netvlad: 908.4698813297736
transvlad: 1.421212358473558
gem: 795.8761735495427
transvpr: 844.0984215040182
ablation: 6.639418441854087

idx: 1059
convap: 611.168471598113
cosplace: 1040.7628097159734
netvlad: 1001.9455648235431
transvlad: 15.79698620763922
gem: 1019.7835198244361
transvpr: 532.8452529173588
ablation: 1131.1802905489435

idx: 3755
convap: 753.6789304537366
cosplace: 717.6598039557682
netvlad: 623.2801361725386
transvlad: 2.1021501953741524
gem: 613.9206752350133
transvpr: 684.3359337978756
ablation: 723.8439573686582

idx: 3833
convap: 716.1953285367314
cosplace: 615.3754482429022
netvlad: 735.5838716437474
transvlad: 1.2307040535982205
gem: 734.4413657333922
transvpr: 613.3480862935911
ablation: 1.2307040535982205

idx: 3896
convap: 752.5961097337528
cosplace: 630.0859421158477
netvlad: 667.4878790725822
transvlad: 2.3267103000852605
gem: 634.8811156099387
transvpr: 555.2477164165584
ablation: 1.9752572537276962

idx: 4629
convap: 877.6041316711679
cosplace: 520.3342405390769
netvlad: 755.7391994371936
transvlad: 2.7529171097938088
gem: 746.7218259689827
transvpr: 759.0549446996554
ablation: 1.799919572204983

idx: 6241
convap: 798.7658421089085
cosplace: 1018.4758688008919
netvlad: 888.7235274675139
transvlad: 1.1308598808612387
gem: 1033.3314547697253
transvpr: 896.0785196484059
ablation: 1.819656517681655

idx: 9861
convap: 250.12050603625408
cosplace: 488.65105453937974
netvlad: 23.400185430783836
transvlad: 1085.7749298277688
gem: 14.706518816949698
transvpr: 20.104722614032262
ablation: 428.49096959089894

idx: 6646
convap: 443.68244707732754
cosplace: 1.6940394951431033
netvlad: 4.705700528900643
transvlad: 618.5668109622071
gem: 68.80035650066964
transvpr: 6.465642121836962
ablation: 4.705700528900643
'''

image = 'front'

result_prefix = './eval/dim_ex'

# methods = ['convap', 'cosplace', 'netvlad', 'transvlad']
methods = [f'{result_prefix}/{i}1280_{image}_512.txt' for i in ['convap', 'cosplace', 'netvlad', 'transvlad']]
methods.append(f'{result_prefix}/gem1280_{image}_2048.txt')
methods.append(f'{result_prefix}/transvpr1280_{image}_256.txt')
methods.append(f'{result_prefix}/transvlad1280_{image}_512_without_mlp_mixer.txt')

save_dir_prefix = './qualitative'

img_dir = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat'
img_list = sorted(list(Path(img_dir).glob('*.png')))

def txt_reader(dir):
    result = []

    with open(dir, 'r') as file:
        for line in file:
            if line[0] == '#': continue

            result.append(float(line.split(' ')[0]))

    return result
    

def check():

    convap_errors_list = txt_reader(methods[0])
    cosplace_errors_list = txt_reader(methods[1])
    netvlad_errors_list = txt_reader(methods[2])
    transvlad_errors_list = txt_reader(methods[3])
    gem_errors_list = txt_reader(methods[4])
    transvpr_errors_list = txt_reader(methods[5])
    ablation_errors_list = txt_reader(methods[6])

    cnt = 0
    th = 500 # m
    idx = 0

    for convap, cosplace, netvlad, transvlad, gem, transvpr, ablation \
         in zip(convap_errors_list, \
            cosplace_errors_list, \
            netvlad_errors_list,\
            transvlad_errors_list,\
            gem_errors_list,\
            transvpr_errors_list,\
            ablation_errors_list):

        condition = not (convap > th) and \
                    (cosplace > th) and \
                    not (netvlad > th) and \
                    not (transvlad > th) and \
                    not (gem > th) and \
                    not (transvpr > th) and \
                    not (ablation > th)
        
        if condition:

            print('=======================================')

            print(f'idx: {idx}')
            print(f'convap: {convap}')
            print(f'cosplace: {cosplace}')
            print(f'netvlad: {netvlad}')
            print(f'transvlad: {transvlad}')
            print(f'gem: {gem}')
            print(f'transvpr: {transvpr}')
            print(f'ablation: {ablation}')
            cnt+=1

        idx += 1

    print(cnt)


    # img = Image.open(img_list[index]).convert('RGB')

    # img.save('normalize_result/vertical_before_normalize.png')

def save_image():
    pass

if __name__ == '__main__':
    check()