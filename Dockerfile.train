FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y wget unzip python3 python3-pip htop

RUN pip3 install python-etcd 

ADD requirements.txt . 
RUN pip3 install -r requirements.txt

RUN mkdir -p /workspace/
ADD train_multi.py /workspace/train_multi.py
ADD nlp_model/ /workspace/nlp_model/

