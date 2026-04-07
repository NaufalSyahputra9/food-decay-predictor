FROM python:3.11-slim AS builder 

WORKDIR /build 
COPY requirements.txt . 
RUN pip install --no-cache-dir --user -r requirements.txt 
FROM python:3.11-slim AS runner 

WORKDIR /app 
COPY --from=builder /root/.local /root/.local 
COPY . . 

ENV PATH=/root/.local/bin:$PATH 
ENV PYTHONUNBUFFERED=1 
ENV DATABASE_URL=sqlite:////app/data/shelflife.db 
RUN mkdir -p /app/data # pastikan folder data ada 
EXPOSE 7860 
CMD ["python", "app/app.py"]