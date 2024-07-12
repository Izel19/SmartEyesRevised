FROM python:3.10

ADD Backend.py .

RUN python -m pip install --upgrade pip

RUN pip install ultralytics fast_colorthief supervision opencv-python setuptools numpy==1.23.5

CMD ["python", "./Backend.py"]
