FROM python:3.12.2
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
COPY . .
EXPOSE 8000
WORKDIR ./MyprojectChatBot
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
