# Object Detection PepsiCo Product

## Introduction

- This is a program for detecting PepsiCo products. The program uses a pretrained model to identify products in images.

## Folder Structure

The directory contains the following parts:

- **ipynb_files**: Contains Jupyter Notebook files used for model training.
- **webapp**: Contains the web application for deploying the program.
  - **static**: Folder that contains static files like images.
  - **templates**: Contains HTML templates for the web application, including `home.html` and `image.html`.
- **weights**: Contains the model's training weight files.
- **app.py**: Source code for the web application.
- **requirements.txt**: A list of required modules and libraries to run the program.

## Installation

- To install the program, follow these steps:
  - Open a terminal or command prompt.
  - Navigate to the directory containing the program's source code using the `cd path/to/your/folder` command.
  - Run the following command to install the required modules and libraries:

```
pip install -r requirements.txt
```
## Running the Program

- To run the program, after installing the required modules, follow these steps:
  - Open a terminal or command prompt.
  - Navigate to the directory containing the program's source code using the cd path/to/your/folder command.
- Run the web application using the following command:
```
python app.py
```
- You can then access the web application at http://localhost:5000 in a web browser.

## Contact

If you have any questions or need support, please contact via email: damkhoaanh@gmail.com.