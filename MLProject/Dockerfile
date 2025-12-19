FROM python:3.12-slim


WORKDIR /app


RUN pip install pandas scikit-learn mlflow


COPY . /app


CMD ["python", "modelling.py"]