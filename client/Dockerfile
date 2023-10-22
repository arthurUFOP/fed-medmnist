FROM tensorflow/tensorflow

WORKDIR /client

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "./client.py"]