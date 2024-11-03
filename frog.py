import numpy as np

def main():
    a = '/media/moon/T7 Shield/master_research/transvlad5_oxford_0828.npy'
    aa = '/media/moon/T7 Shield/master_research/transvlad5_oxford_0519.npy'

    # (18966, 512)
    b = np.load(a)
    bb = np.load(aa)


    cnt = 0

    for i in range(b.shape[0]):
        if sum(b[i, :]) == 0.0:
            cnt +=1

    print(cnt)


if __name__ == '__main__':
    main()