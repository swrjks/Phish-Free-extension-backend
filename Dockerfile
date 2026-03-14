FROM python:3.10-slim
WORKDIR /app
COPY requirements_phase1-3.txt .
RUN pip install --no-cache-dir -r requirements_phase1-3.txt
COPY . /app
ENV FLASK_APP=app.py
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
