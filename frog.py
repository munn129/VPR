from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import torchvision.transforms as tvf

transform = tvf.Compose(
    [tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
     tvf.ToTensor(),
     tvf.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])]
)

def main():

    index = 5000

    img_dir = '/media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat'
    img_list = sorted(list(Path(img_dir).glob('*.png')))

    img = Image.open(img_list[index]).convert('RGB')

    th = int(img.size[0]/2)
    f = img.crop((0,0,th,th))
    r = img.crop((th,0,th * 2,th))
    l = img.crop((0, th, th, th * 2))
    b = img.crop((th, th, th * 2, th * 2))

    # horizontal
    # img = Image.new("RGB", (th * 4, th))
    # img.paste(f, (0,0))
    # img.paste(r, (th,0))
    # img.paste(l, (th * 2,0))
    # img.paste(b, (th * 3,0))

    # vertical
    img = Image.new("RGB", (th, th*4))
    img.paste(f, (0,0))
    img.paste(r, (0,th))
    img.paste(l, (0,th * 2))
    img.paste(b, (0,th * 3))

    img.save('normalize_result/vertical_before_normalize.png')
    to_pil_image(transform(img)).save('normalize_result/vertical_after_normalize.png')

if __name__ == '__main__':
    main()