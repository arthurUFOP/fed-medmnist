FROM tensorflow/tensorflow

WORKDIR /server

COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "./server.py"]