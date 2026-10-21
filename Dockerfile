FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Stand-in data so the container works out of the box; mount real data over /app/data/cs-train
RUN python generate_sample_data.py && python run_tests.py
EXPOSE 8080
ENV PORT=8080
CMD ["python", "app.py"]
