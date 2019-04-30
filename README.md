[//]: # (References)
[Green]: ./images/MobilenetGreen.png
[Yellow]: ./images/MobilenetYellow.png
[Red]: ./images/MobilenetRed.png
[MobilenetLoss1]: ./images/MobilenetLoss1.PNG
[MobilenetLoss2]: ./images/MobilenetLoss2.PNG
[Nodes]: ./images/Nodes.png


### Introduction

This is the documentation for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car prepared individually.

The objective of the project is to let the ego car drive itself around the track, respecting the traffic light regulations and sticking to the current lane by being guided by the waypoint projection. Some of the node scripts were based on the Partial Walkthrough (e.g. dbw_node.py and twist_controller.py) to a large extent.

The detection is based on the SSD Mobilenet Tensorflow model trained in the Windows 10 environment. 

The approach could not be fully tested on the track given the technical time-out issue described below.

### Project Issues

The simulator exhibited very low performance irrespective of the tested approach. This is a known issue reported by manu users on github/slack/knowledge.
1. VM VirtualBox + native Windows 10 simulator - very high latencies even with minimal graphical settings and 10Hz ROS refresh for the nodes. Latancy causing erratic DBW car behaviour and derailing with camera on.
2. VM Virtual Box + VM simulator - incomplete display of landscape in simulator irrespective of the graphics setting (controller/acceleration/MB) in the VM. Latancy causing erratic DBW car behaviour with Camera on. Tested on all acceleration settings.
3. Workstation ROS and Simulator - poor simulator performance and latancy causing erratic DBW car behaviour and derailing with camera on.
4. Embedded Ubuntu 18 + ROS Melodic Morenia + native Windows 10 simulator - very poor latency but somewhat better than other approaches. (older Ubuntu distribution with Kinetic was also tested)
The approaches were tested on two 2018 computers (i5, 16GB RAM, AMD and Intel graphics unit). Tested on all graphics settings also on the older versions of the Simulator.

The embedded Windows 10 linux instance with ROS Melodic Morenia was ultimately used. However this approach produced an socket timeout error reported here (also experienced by other users) after approximately 1 minute of car driving itself. I was not able to address this issue.
[Udacity Knowledge Issue](https://knowledge.udacity.com/questions/37234).
[Waffle Board Issue Report](https://waffle.io/udacity/sdc-issue-reports/cards/5ca59d73bc8733010da92f02).

```
[styx_server-2] process has died [pid 5309, exit code 1, cmd /home/ubuntu/Project2/CarND-Capstone/ros/src/styx/server.py __name:=styx_server __log:=/home/ubuntu/.ros/log/cf567134-55fd-11e9-99eb-e4e7491b8d72/styx_server-2.log].
log file: /home/ubuntu/.ros/log/cf567134-55fd-11e9-99eb-e4e7491b8d72/styx_server-2*.log
```

Given current version of Tensorflow was used, the Tensorflow in the Workspace hast to be updated by running

```
pip install --upgrade tensorflow
```
(also included in the requirements.txt file):
```
pip install -r requirements.txt
```

### Ubuntu and ROS Installation

Ubuntu download [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:  2 CPU  2 GB system memory 25 GB of free hard drive space

ROS installation
  * [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) if you have Ubuntu 18.
  * [ROS Kinetic](http://wiki.ros.org/melodic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.

[Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

Updating the keys for ROS installation.

```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

```
If it producess and error, manual download should be used:
```
curl -sL "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x421C365BD9FF1F717815A3895523BAEEB01FA116" | sudo apt-key add
```

Installation
```
sudo apt update
sudo apt install ros-melodic-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
```

Additional ROS/Python dependencies for the nodes to run properly
```
sudo apt-get install -y ros-melodic-dbw-mkz-msgs
rosdep install --from-paths src --ignore-src --rosdistro=melodic -y
sudo apt install python-pip
pip install eventlet
pip install scipy
pip install flask
pip install attrdict
pip install pathlib
pip install flask-socketio
pip install tensorflow
```

### Workspace Installation

Given current version of Tensorflow was used, the Tensorflow in the Workspace hast to be updated by running

```
pip install --upgrade tensorflow
```
(also included in the requirements.txt file):
```
pip install -r requirements.txt
```

### Traffic Light Model Training Assumption

The following resources were used:
1. Pre-trained model from the [Tensorflow API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Given poor performance of communication between the ROS and Simulator, the fastest models were tested for the purpose of the project, based on the benchmarks provided under the link above.
- ssd_inception_v2_coco_2018_01_28
- ssd_mobilenet_v1_coco_2018_01_28 [Link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).

The initial image set was exported within the get_light_state() method in the tl_detector.py script. However, given the time consuming nature of manual labeling and the fact that the captured screens were similar, I looked for suitable online datasets with traffic light. The following sources were used:

1. Annotated extract of the Bosh traffic light dataset - random 440MB subsample given 6GB+ size of the dataset converted to .record format (e-mail has to be provided for the data link to be generated)
https://hci.iwr.uni-heidelberg.de/node/6132

The conversion of the Bosh dataset to the TFrecord was performed by customizing the following python scripts:
- [Tensorflow data conversion script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py).
- [Oleg Leizerov script](https://github.com/olegleyz/traffic-light-detection/blob/master/bosch-to-tfrecords.py).

2. Annotated extract of simulator screen data (thanks to the effor of Vatsal Srivastava)
https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view

Given that the performance of the model with only Bosh and Vatsal dataset on the REAL Udacity images was very limited (only the simulator screens were reliably detected), another dataset was added to the shuffled bundle:

3. Annotated dataset prepared by Alex Lechner
https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0

In total, the input data structure was the following:
- Bosh dataset 440MB 
- Udacity simulator oversampled dataset 160MB
- Udacity real car sample 50MB

The training process was performed:
- for all data in the first 22000 steps
- on simulator images ONLY in the next 2000 steps, since it was observed that the model loses ability to generalize in the simulator when the training data is mixed with the Bosh dataset.
- for all data in the following 6000 steps

### Tensorflow API Installation and Training

The model was trained locally on the Windows 10 machine. Installation of the Tensoflow API is a prerequisite. Anaconda with Term 1 environment was used. [CARND Term 1 Installation Instructions](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)

1. Clone the Tensorflow model repository and configure CARND Term 1
```
git clone https://github.com/tensorflow/models
```

2. Download and place the model of choice in the models\research\object_detection\models\ folder
[SSD Mobilenet](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) was used given marginally faster classification performance over the SSD Inception model.

3. Install dependencies

```
pip install pip --upgrade
pip install tensorflow --upgrade
```

4. Download and set the [Protoc](https://github.com/google/protobuf/releases) paths

(on Linux)
```
protoc object_detection/protos/*.proto --python_out=.
```

(on Windows)
```
protoc object_detection/protos/anchor_generator.proto --python_out=.
protoc object_detection/protos/argmax_matcher.proto --python_out=.
protoc object_detection/protos/bipartite_matcher.proto --python_out=.
protoc object_detection/protos/box_coder.proto --python_out=.
protoc object_detection/protos/box_predictor.proto --python_out=.
protoc object_detection/protos/eval.proto --python_out=.
protoc object_detection/protos/faster_rcnn.proto --python_out=.
protoc object_detection/protos/faster_rcnn_box_coder.proto --python_out=.
protoc object_detection/protos/grid_anchor_generator.proto --python_out=.
protoc object_detection/protos/hyperparams.proto --python_out=.
protoc object_detection/protos/image_resizer.proto --python_out=.
protoc object_detection/protos/input_reader.proto --python_out=.
protoc object_detection/protos/keypoint_box_coder.proto --python_out=.
protoc object_detection/protos/losses.proto --python_out=.
protoc object_detection/protos/matcher.proto --python_out=.
protoc object_detection/protos/mean_stddev_box_coder.proto --python_out=.
protoc object_detection/protos/model.proto --python_out=.
protoc object_detection/protos/multiscale_anchor_generator.proto --python_out=.
protoc object_detection/protos/optimizer.proto --python_out=.
protoc object_detection/protos/pipeline.proto --python_out=.
protoc object_detection/protos/post_processing.proto --python_out=.
protoc object_detection/protos/preprocessor.proto --python_out=.
protoc object_detection/protos/region_similarity_calculator.proto --python_out=.
protoc object_detection/protos/square_box_coder.proto --python_out=.
protoc object_detection/protos/ssd.proto --python_out=.
protoc object_detection/protos/ssd_anchor_generator.proto --python_out=.
protoc object_detection/protos/string_int_label_map.proto --python_out=.
protoc object_detection/protos/train.proto --python_out=.
```

(alternatively)
```
protoc/bin/protoc object_detection/protos/*.proto --python_out=.
```

4. Activate the environment
```
activate carnd-term1
```

5. Set the variables (update the full system path)

```
set PYTHONPATH=%PYTHONPATH%;YOURPATH\models\research
set PYTHONPATH=%PYTHONPATH%;YOURPATH\models\research\slim
```

The pipeline configuration file is provided in the Workspace under CarND-Capstone/model/. The key customization that improves performance is the decrease of the max_detections_per_class and max_total_detections to 4.

Training the model was performed with legacy train.py due to the Windoes 10 Visual Studio version problem with pycocotools building. 
```
python models/research/object_detection/legacy/train.py --logtostderr --train_dir=models/research/object_detection/models/ --pipeline_config_path=models/research/object_detection/models/ssd_mobilenet_v1_coco_2018_01_28/myPipeline.config
```

For SSD Inception a new pipeline config file was used, given known bug in the original one. [New config for inception](https://github.com/developmentseed/label-maker/blob/94f1863945c47e1b69fe0d6d575caa0b42aa8d63/examples/utils/ssd_inception_v2_coco.config)

The loss plots can be observed by using the following command and pasting the generated address to the browser on the local machine.
```
tensorboard --logdir=models/research/object_detection/models
```

First 2000 steps

![Mobilenet Loss][MobilenetLoss1]

Another 2000 steps

![Mobilenet Loss][MobilenetLoss2]

Important consideration: The message content for the traffic light has different integer coding comparing to the labels, given that 0 shouldn't be used in the pbtxt label map file. 

ROS message:

```
uint8 UNKNOWN=4
uint8 GREEN=2
uint8 YELLOW=1
uint8 RED=0
```
Label file (label_map.pbtxt) structure: Green (1) Red (2) Yellow (3) Unknown (4)

Exporting the frozen graph for the model

```
python models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path models/research/object_detection/models/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix models/research/object_detection/models/model.ckpt-4000 --output_directory export_mobilenet
```

The output of the training process is the model graph in the form of frozen_inference_graph.pb file which can be directly used in the tl_classifier.py script.

After the training the model was able to classify the traffic lights with high degree of confidence.

![Green][Green]

![Yellow][Yellow]

![Red][Red]

### Node Architecture

The project is based on the set of nodes that work together to produce a predictable car behaviour. The map of Nodes and corresponding is presented below (source: Udacity):

![Nodes][Nodes]

Key nodes are:
1. Traffic Light Detection consisting of scripts
- tl_detector.py which processes the images and manages subscribing and publishing actions.
- tl_classification.py containing the neural network models described in the Traffic Light Model Training Assumption and Tensorflow API Installation and Training sections of this readme.

2. Waypoint Management consisting of script:
- waypoint_updater.py which loads the waypoints and controls proximity of the stoplines.

3. Drive By Wire node consisting of scripts:
- dbw_node.py which controls car's behaviour.
- twist_controller.py which contains the parameters of the PID controller and measures to stop the vehicle.

Other nodes were not modified (e.g. Waypoint Follower implementing the pure pursuit in CPP).

### Performance Improvement

The following measures were applied to enhance the performance of the solution:
- image recognition is applied every i'th image received (driven by counter_processing variable in the tl_detector.py)
- traffic light detection happens only at the distance of 300 or less to the next stopline (set as a constant in the TL_DIST variable in the tl_detector.py)
- reducing max_detections_per_class and max_total_detections to 4 in the pipeline.config file for model training.

### Port Forwarding
To set up port forwarding for VM, forllow [the instructions](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository.
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Copy the SRC folder from the Workspace

3. Install python dependencies.
```bash
cd CarND-Capstone
pip install -r requirements.txt
pip install --upgrade tensorflow
```
4. Make and run styx.
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
5. Run the simulator.

6. Wait for the 'Entire model loaded' message in the console and release Manual and Camera buttons.