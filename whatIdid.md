# Plan Jupyter Notebook Sanbox: 
## The Automated "Cat vs. Dog" Classifier (CI/CD Focus)
Instead of just training a model in a Jupyter Notebook, build a pipeline that automatically tests and deploys your model every time you update your code.

The ML Part: A basic image classifier using a pre-trained model (like MobileNet) to distinguish between cats and dogs.

The DevOps Part: * Use GitHub to store your code.

Set up GitHub Actions so that whenever you push a change, it automatically runs tests (e.g., checking if the model outputs the right data shape)

If the tests pass, use GitHub Actions to build a Docker image of your model and push it to Docker Hub.
Why it’s great: It teaches you the absolute core of DevOps: Continuous Integration and Continuous Deployment (CI/CD).

## What am I gonna do?
1. collect raw data 
2. preprocess it
3. tune it
4. train model -> check accuracy 
5. then I'll move onto the DevOps part

## Expected Folder Structure:

```text
cat_dog_project/
├── .github/
│   └── workflows/              # GitHub Actions CI/CD configuration files
│       └── cicd.yml
├── data/
│   ├── raw/                    # Where you download the original dataset
│   └── processed/              # Data split into train/validation sets
├── src/
│   ├── __init__.py
│   ├── train.py                # Script to train the model
│   └── evaluate.py             # Script to test accuracy and model outputs
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI production server code
├── tests/
│   ├── __init__.py
│   └── test_app.py             # Automated unit tests for CI/CD
├── models/                     # Directory to store your trained .pth file
├── Dockerfile                  # Container configurations
├── requirements.txt            # Project python dependencies
└── .gitignore
```


## Preprocessing
1. get to file path
2. check total data we have in cat and dog folder
3. remove zero-byte & corrupted non jpeg files
4. now clean and store data
5. do train test split using sklearn

### Thought: 01
1. trian_paths « contains 80% chunk of image paths i.e. ~20k images
2. val_paths « remaining 20% is used to test the model
3. train_labels and val_labels « contains encoded labels i.e. 0 & 1 

Finally, at this point my data is correctly split, perfectly balanced and isolated.

1. Before creating a pytorch dataset, I need to define how to transform(MobileNet) raw images with colors normalized to a specific scale.

So, now I have a few questions in mind:
1. what do I need pytorch for? « to load mobileNet and feed our pixels into it to get a prediction out 
2. what exactly is mobileNet? « A CNN developed by Google which already contains multiple images of different objects, and it recognizes how to differentiate between them
3. why can't I use the train test data right now for checking model accuracy? « because my machine can't see the image like humans do, so I will need something that can detect edges on ears, textures, shapes and help me recognize the object in my image



##### Creating custom dataset class in pytorch and passing the transformation recipie to the dataset
##### Now lets create DataLoaders of batchsize = 32(standard lightweight size), we will feed 32 images at a time into the model
##### Loading the mobileNet model
##### checking out losses and optimizing based on ADAM -> as it is better for momentum
##### training the model resulted in quite a good result