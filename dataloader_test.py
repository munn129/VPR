

class IDPreserveTwoDataset(torch.utils.data.Dataset):
    # image_width = 320
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    condition_channels = 2
    tf_condi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * condition_channels, [0.5] * condition_channels)
    ])

    def __init__(self, image_path_list1, image_path_list2):
        self.image_path_list1 = sorted(p.resolve() for p in Path(image_path_list1).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        self.image_path_list2 = sorted(p.resolve() for p in Path(image_path_list2).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list1)

    def __getitem__(self, index):
        img1 = Image.open(self.image_path_list1[index]).convert('RGB')
        # condi_image = self.tf_real(img1)
        r, g, b = img1.split()
        img_condi = np.stack([b, r], axis=2)  # "r" is same as "g"
        condi_image = self.tf_condi(img_condi)

        img2 = Image.open(self.image_path_list2[index]).convert('RGB')
        target_image = self.tf_real(img2)

        sample = {'condition_image': condi_image, 'real_image': target_image}
        return sample

train_dataset = datasets.IDPreserveTwoDataset(args.data_dir+'/condi',args.data_dir+'/orig')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=args.workers)