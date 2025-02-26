## AdaGrasp: A Stiffness and Grasp Affordance Dataset and a Transformer-Based Adaptive Grasp Method

PyTorch implementation of paper "AdaGrasp: A Stiffness and Grasp Affordance Dataset and a Transformer-Based Adaptive Grasp Method"

## Visualization of the architecture
<img src="img/grasp-transformer.png" width="500" align="middle"/>
<br>

This code was developed with Python 3.8 on Ubuntu 18.04.  

## Datasets

Currently, both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php),
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/) , and [GraspNet 1Billion](https://graspnet.net/datasets.html)  are supported.

### Cornell Grasping Dataset
1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 

### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

### GraspNet 1Billion dataset

1. The dataset can be downloaded [here](https://graspnet.net/datasets.html).
2. Install graspnetAPI following [here](https://graspnetapi.readthedocs.io/en/latest/install.html#install-api).

   ```bash
   pip install graspnetAPI
   ```   
3.  We use the setting in [here](https://github.com/ryanreadbooks/Modified-GGCNN) 


## Training

Training is done by the `main.py` script.  

Some basic examples:

```bash
# Train  on Cornell Dataset
python main.py   --dataset AdaGrasp
```

Trained models are saved in `output/models` by default, with the validation score appended.

## Visualize
Some basic examples:
```bash
# visulaize grasp rectangles
python visualise_grasp_rectangle.py   --network your network address

# visulaize heatmaps
python visulaize_heatmaps.py  --network your network address

```


## Acknowledgement
Code heavily inspired and modified from https://github.com/IDEA-Research/DINO


