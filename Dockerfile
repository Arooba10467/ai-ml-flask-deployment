FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install flask pillow numpy

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

CMD ["python", "flask_api.py"]
