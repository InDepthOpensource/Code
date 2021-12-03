## How to perform inference and performance benchmarks
To benchmark the latency of our DNN, compile with TensorRT:
- If using Jetson, make sure Jetson is in the MAXN mode (40W, all cores) by using sudo ${HOME}/jetson_clocks.sh (set static max performance for CPU, GPU, EMC clocks)
- cd <InDepth dir>/experiments/
- python3 export_to_onnx.py 
- cd ../tmp/
- trtexec --onnx=depth_completion.onnx --saveEngine=depth_completionfp16.trt --explicitBatch --fp16 --inputIOFormats=fp16:chw,fp16:chw --workspace=128

Running the eval_models.py script would give worse than reported latency performance, as it doesn’t use tensorRT. 

## How to setup and run PyTorch DNN training and evaluation code
1. Get Matterport3D dataset. As described on their website (https://niessner.github.io/Matterport/), Matterport3D dataset requires you to sign an user agreement first, and then email one of the dataset maintainers to get access.
2. After you get access to Matterport3D dataset, email Yinda Zhang (yindaz (at) gmail (dot) com, https://www.zhangyinda.com, https://github.com/yindaz/DeepCompletionRelease/tree/master/data) to access the Matterport3D reconstructed depth ground truth and Matterport3D reconstructed normals.
3. Download this code and install the dependencies. If some of the dependencies are still missing please make your best judgement and install them. Also, please unzip the `file_lists.zip` under `./tmp/` folder, and then copy the two acquired txt files **directly** into `./tmp/` folder (these files are zipped only to avoid GitHub upload limit). Finally, our pretrained model is available at (https://drive.google.com/file/d/10UbsmaS9ZgvTsKTtnY67TuFrVPa1llhf/view?usp=sharing). Please download this model, and then put it into `./tmp/` folder to use it.
4. Unzip the Matterport3D dataset and the ground truth you obtained from Yinda Zhang. Then, please refer to the format in `./tmp/train_file_list_normal.txt` and `./tmp/train_file_list_normal.txt` and write the absolute paths of **your** unzipped Matterport3D files into these file lists. This is important as the training and testing code will read these file lists and then use the listed files for training and testing. To ensure fairness to previous works in depth completion, please make sure you are using the same training and testing sequence as described in `./tmp/train_file_list_normal.txt` and `./tmp/train_file_list_normal.txt`.
5. Then, you can evaluate/finetune the provided model, or train a new model on your own. Please notice this code will only run on CUDA enabled computers. Also, to train or finetune our model, more than 8 GB of VRAM should be available.
  - To evaluate the model, please `cd` into the `./experiments/` folder, and then run `python3 eval_models.py`. The script will report the metrics and then save the result as pictures into the same folder.
  - To finetune the model, please run `python3 train.py --load-model=../tmp/best_l1.pth`
  - To train the model from scratch, please run `python3 train.py`

## Code
Contains the PyTorch code for training and evaluating DNN described in InDepth IMWUT submission. Pretrained weights are also available (see the PyTorch setup section above). There are also some scripts for post-processing the ToF18K dataset. 

## Dataset
The ToF18K dataset contains 18.6 K RGB-D images captured in indoor environments such as apartments, classrooms, lounges, office spaces, and university labs. It is captured with the RGB and ToF camera on a Samsung Galaxy Note 10+ phone. The dataset will be made publically available once the paper is accepted. Meanwhile, some sample images from the dataset are shown below. 

![2020-11-05T18:40:381366_colorized_projected](https://user-images.githubusercontent.com/85352183/144230500-f1cb1fb4-aff5-498b-be7a-660924d3b67d.png)
![2020-11-05T18:40:381366_undistorted](https://user-images.githubusercontent.com/85352183/144230503-b06889b5-4672-456c-b8bd-3d2a19a4c446.jpg)
The image above shows a RGB-D image captured in a university lab. Cooler colors refer to objects and surfaces that are closer and warmer colors refer to surfaces that are further away. There are regions with large depth values on the right side of the image, due to depth camera artifacts mentioned in the paper. Also note the depth values for floors, ceilings, and oblique surfaces are missing as consumer ToF cameras cannot effectively capture surfaces that are parallel to the optical axis of the camera. 

![2020-11-05T18:49:05107_colorized_projected](https://user-images.githubusercontent.com/85352183/144230596-98814705-c5a0-41d3-b947-e37800801105.png)
![2020-11-05T18:49:05107_undistorted](https://user-images.githubusercontent.com/85352183/144230598-418ba9d2-a77c-4121-ac19-2c19bc26fcf4.jpg)
The image above shows a whiteboard on the wall and a chair. Note the right side of the depth image contains artifacts similar to image above.  
  
![2020-11-06T14:43:4418_colorized_projected](https://user-images.githubusercontent.com/85352183/144230647-43457b3e-42ad-45b5-95d1-e07bb4475224.png)
![2020-11-06T14:43:4418_undistorted](https://user-images.githubusercontent.com/85352183/144230648-de66d501-135f-420a-af2d-4f23bafa0f99.jpg)
The image is taken inside an apartment. Again, there are artifacts on the right side of the depth image. Also note the floor and the desk is missing in the depth as it is too dark. 

![2020-10-09T12:57:490_colorized_projected](https://user-images.githubusercontent.com/85352183/144230740-27fba7e8-bed2-4644-beda-161b961c0dd9.png)
![2020-10-09T12:57:490_undistorted](https://user-images.githubusercontent.com/85352183/144230742-2326797b-7c62-4e91-bfd6-8065dd42711e.jpg)
The image is taken in an office. There is no significant depth artifact on this image. However, the depth camera cannot capture surfaces that are too bright or too dark, such as the window, the keyboard, the mouse, and the computer case in the image. 

