

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Xinyu-ham/nlp-pipeline">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Fake News Detection from Training to Deployment</h1>

  <p align="center">
    complete end-to-end MLOps pipeline for a simple multimodal Machine Learning project.
    <br />
    <a href="#introduction"><strong>Start Here! »</strong></a>
    <br />
    <br />
    <a href="#installation">View Demo</a>
    ·
    <a href="https://github.com/Xinyu-ham/nlp-pipeline/issues">Report Bug</a>
    ·
    <a href="https://github.com/Xinyu-ham/nlp-pipeline/issues">Request Feature</a>
  </p>
</div>

## Table of Contents
- [End-to-End MLOps pipeline](#end-to-end-mlops-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction
The purpose of this project is to develop a complete, industrial grade Machine Learning system. The project is based on a simple multimodal Machine Learning project, which is to detect fake news from news headlines. This project is designed to be a complete end-to-end MLOps pipeline, which ensures continuous integration, continuous delivery and continous training of a Machine Learning system. The project is also designed to be scalable and reproducible, which means it can be easily deployed to a cloud platform and can be easily reproduced by other developers. 

This project will include not only the finished code, but will also document the process of developing the project. The project will be developed in an iterative and modular way, where we explore different modern technologies that are designed to help ML practitioners to develop a robust and scalable ML system.


### A complete end-to-end MLOps pipeline


Start local distributed run:
```sh
torchrun --standalone --nnodes=1 --nproc-per-node=gpu train_multi.py
```
Start local MLFlow server:
```sh
mlflow server --default-artifact-root s3://xy-mp-pipeline/ --artifacts-destination s3://xy-mp-pipeline/mlflow
```
