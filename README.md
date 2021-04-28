# cortx_pytorch
Fast interface between pytorch and Segate CORTX

This package let you encode and upload a pytorch computer vision dataset (of the shape (image,label)) to CORTX.

## 1: Install

```
pip install cortx_pytorch
```

## 2: Convert and upload your dataset

```python
from cortx_pytorch import upload_cv_dataset, make_client
from torchvision import datasets

if __name__ == '__main__':
    # Define the connection settings for our client
    client = make_client(URL, ACCESS_KEY, SECRET_KEY)


    bucket = 'testbucket'  # Bucket where to read/write our ML dataset
    folder = 'imagenet-val'  # Folder where this particular dataset will be

    # We use a pytorch dataset as a source to prime the content of CORTX
    # Once we have encoded and uploaded it we don't need it anymore
    # Here we use a locally available Imagenet dataset
    ds = ds = datasets.ImageFolder('/scratch/datasets/imagenet-pytorch/val')

    # Packs and upload any computer vision dataset on cortx
    #
    # It only needs to be done once !
    # Image are groupped in objects of size at most `masize` and at most
    # `maxcount` images. We use `workers` processes to prepare the data
    # in parallel
    upload_cv_dataset(ds, client=client, bucket=bucket,
                      base_folder=folder, maxsize=1e8,
                      maxcount=100000, workers=30
```

## 2: Use the dataset like any pytorch dataset

```python
fimport torch as ch
from tqdm import tqdm

from cortx_pytorch import RemoteDataset, make_client
from torchvision import transforms

preproc = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
        
if __name__ == '__main__':

    # Define the connection settings for our client
    client = make_client(URL, ACCESS_KEY, SECRET_KEY)

    bucket = 'testbucket'  # Bucket where to read/write our ML dataset
    folder = 'imagenet-val'  # Folder where this particular dataset will be
    
    # Now that we have created and upload the dataset on CORTX we can use
    # it in Pytorch
    dataset = (RemoteDataset(client, bucket, folder)
        .decode("pil") # Decode the data as PIL images
        .to_tuple("jpg;png", "cls") # Extract images and labels from the dataset
        .map_tuple(preproc, lambda x: x) # Apply data augmentations
        .batched(64)  # Make batches of 64 images
    )
    # We create a regular pytorch data loader as we would do for regular data sets
    dataloader = ch.utils.data.DataLoader(dataset, num_workers=3, batch_size=None)
    for image, label in tqdm((x for x in dataloader), total = 100000 / 60):
        # Train / evaluate ML models on this batch of data
        pass

```
