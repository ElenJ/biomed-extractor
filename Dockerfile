FROM python:3.11

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set NLTK data path and download 'punkt'
ENV NLTK_DATA=/usr/share/nltk_data
RUN mkdir -p /usr/share/nltk_data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/share/nltk_data')"


COPY app app

CMD ["streamlit", "run", "app/main.py"]