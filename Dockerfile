FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements dan install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Beri izin eksekusi ke start.sh
RUN chmod +x start.sh

# Port publik
EXPOSE 7860

# Jalankan script
CMD ["./start.sh"]