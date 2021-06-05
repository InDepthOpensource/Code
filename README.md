<<<<<<< HEAD
## How to setup and run this code
1. Get Matterport3D dataset. As described on their website (https://niessner.github.io/Matterport/), Matterport3D dataset requires you to sign an user agreement first, and then email one of the dataset maintainers to get access.
2. After you get access to Matterport3D dataset, email Yinda Zhang (yindaz (at) gmail (dot) com, https://www.zhangyinda.com, https://github.com/yindaz/DeepCompletionRelease/tree/master/data) to access the Matterport3D reconstructed depth ground truth and Matterport3D reconstructed normals.
3. Download this code and install the dependencies. If some of the dependencies are still missing please make your best judgement and install them. Also, please unzip the `file_lists.zip` under `./tmp/` folder, and then copy the two acquired txt files **directly** into `./tmp/` folder (these files are zipped only to avoid GitHub upload limit). Finally, our pretrained model is available at (https://drive.google.com/file/d/10UbsmaS9ZgvTsKTtnY67TuFrVPa1llhf/view?usp=sharing). Please download this model, and then put it into `./tmp/` folder to use it.
4. Unzip the Matterport3D dataset and the ground truth you obtained from Yinda Zhang. Then, please refer to the format in `./tmp/train_file_list_normal.txt` and `./tmp/train_file_list_normal.txt` and write the absolute paths of **your** unzipped Matterport3D files into these file lists. This is important as the training and testing code will read these file lists and then use the listed files for training and testing. To ensure fairness to previous works in depth completion, please make sure you are using the same training and testing sequence as described in `./tmp/train_file_list_normal.txt` and `./tmp/train_file_list_normal.txt`.
5. Then, you can evaluate/finetune the provided model, or train a new model on your own. Please notice this code will only run on CUDA enabled computers. Also, to train or finetune our model, more than 8 GB of VRAM should be available.
  - To evaluate the model, please `cd` into the `./experiments/` folder, and then run `python3 eval_models.py`. The script will report the metrics and then save the result as pictures into the same folder.
  - To finetune the model, please run `python3 train.py --load-model=../tmp/best_l1.pth`
  - To train the model from scratch, please run `python3 train.py`
=======
# Code
Code and dataset for InDepth Sensys 2020 submission. 
>>>>>>> parent of 7b49c15 (Update README.md)
