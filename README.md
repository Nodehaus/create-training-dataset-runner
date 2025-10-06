# Our runner to create training and evaluation datasets

This repository contains a script to create datasets from job files on S3. It also contains a Dockerfile to create a
Ollama severless instance on Runpod to generate the dataset examples via a self-hosted LLM. As alternative you can also
use the OpenAI API with an API key.

## Environment variables

Put them in a `.env` file:

```
APP_ENV=local
AWS_ACCESS_KEY=
AWS_SECRET_KEY=
RUNPOD_API_KEY=
RUNPOD_POD_ID=
APP_S3_BUCKET=nodehaus
AI_PLATFORM_API_BASE_URL=http://localhost:8080
AI_PLATFORM_API_KEY=VerySecureKey
```
