# PointPSSN(Point Pothole-Specialized Segmentation Network)

![image-20250205221322061](https://badpicturebed.oss-cn-hangzhou.aliyuncs.com/img/image-20250205221322061.png)

A deep learning model for pothole semantic segmentation.

## Usage

### Get the project

```bash
git clone https://github.com/CountCarMaster/PointPSSN.git
```

### Download the dataset

Download the dataset from [Google Drive](https://drive.google.com/file/d/1zmQ3IzpPbuiBKXMo-ttX1bqew8GSZ9to/view?usp=sharing). Put it in the root direction of the project and unzip it.

### [Optional] Create a conda environment

```bash
conda create -n PointPSSN python=3.8
conda activate PointPSSN
```

### Install packages

```bash
pip install -r requirements.txt
```

### Train or test!

```bash
python main.py
```

If you want to change the mode (train/test) or change the model, please modify the configuration file `config.yaml` in the folder `src`.

