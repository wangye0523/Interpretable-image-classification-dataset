# Interpretable-image-classification-dataset
We propose an interpretable image classification dataset leveraged from the COCO dataset, with multi-granularity semantic labels. Each image in our dataset has not only its coarse-grained category label, but also the fine-grained semantic label in this image. <br><br>
The fine-grained semantic attributes of our dataset are: person, bicycle, car, motorbike, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suit, frisbee, ski, snowboard, sports ball, kit, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hotdog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, television, laptop, mouse, remote, keyboard, cellphone, microwave, oven, sink, refrigerator, book, clock, vase, scissors, teddy bear, hairdryer and toothbrush, with a total of 80 classes. <br><br>
The coarse-grained category attributes of our dataset are: transport, outdoor sign, animal, accessory, sports, dining tools, furniture decoration, indoor supply, electronic, appliance, and food, with a total of 11 classes. <br><br>
The numbers from 1 to 80 in fine_label.txt correspond to the above fine-grained semantic attributes in turn. And the numbers from 1 to 11 in coarse_label.txt correspond to the above coarse-grained category attributes in turn. <br><br>
 # Dataset Download
 The download link of our dataset is as follows: https://pan.baidu.com/s/1ES1_6FQJWfpcJeEVyb5-Mw, and its extraction code is t5nn. It contains training set and test set. Put the downloaded whole folder in the same path as the files downloaded from GitHub.
 # Requirements
 This is my experiment eviroument:
 * python 3.6
 * pytorch 1.6.0
 * pillow
 * scikit-image
 
 # Load dataset
cd your download path <br>
python load_dataset.py --data_root your download path
 
 
