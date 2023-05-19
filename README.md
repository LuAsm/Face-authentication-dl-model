# Face-authentication-dl-model

Capstone project with idea to make deep learning model which recognize people faces. 

All project is create with python and pytorch. 

Full eda and training you can find in folder "nootebooks" or here:
<a href="https://colab.research.google.com/drive/1XoR7gu7Sjf2JWswpjFXVbz-WOhCOKBV2?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

For training i use LFWPeople pytorch dataset, so if you want you can run code easylie.

If you run code first time it's took some time. About 15min to create preproceeded dataset and about 15min to train model on cpu or about 1min on gpu.

To run project just run python and then type "flask run"

Code automaticly generate dataset and model, if you run it not first time it's just load saved dataset and model.

When you run flask, you got ip address, open it in chrome or some other and push "Load Random Image". This button upload 1 random image from test data to model and give us predict. After that press we will see random image and above that image prediction and truht names.

![Start screen](https://www.dropbox.com/s/7reqsfsb15joup1/1.png?dl=0)

![Load Random Image](https://www.dropbox.com/s/17f1k6yqzxt9m5i/2.png?dl=0)

Also i created option to train trained model with your images. When you open flask link push button "Upload Image for training" here pops up window where you can write your name and select your photos. Then upload your image for model testing after pushing "Upload Image for testing" and finally press "Load Test Image" to look what that model predict from your image.

