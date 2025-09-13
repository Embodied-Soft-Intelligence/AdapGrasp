## AdaGrasp: A Stiffness and Grasp Affordance Dataset and a Transformer-Based Adaptive Grasp Method

PyTorch implementation of paper "AdaGrasp: A Stiffness and Grasp Affordance Dataset and a Transformer-Based Adaptive Grasp Method"

## Visualization of the architecture
<img src="https://raw.githubusercontent.com/Embodied-Soft-Intelligence/AdapGrasp/main/demo/model.jpg" width="1000" align="middle"/>
<br>

This code was developed with Python 3.8 on Ubuntu 18.04.  

## Datasets

AdaGrasp Dateset

## Training

Training is done by the `main.py` script.  

Some basic examples:

```bash
# Train  on AdaGrasp Dataset
python main.py   --dataset AdaGrasp
```

Trained models are saved in `output/models` by default, with the validation score appended.

## Visualize
Some basic examples:
```bash
# visulaize grasp rectangles
python visualise_grasp_rectangle.py   --network your network address

```


## Acknowledgement
Code heavily inspired and modified from https://github.com/IDEA-Research/DINO


