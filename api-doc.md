
# Agent Framework API Documentation

This document provides comprehensive documentation for the Agent Framework API endpoints deployed at `https://mira-0gn4.onrender.com`.

## Table of Contents
- `[Authentication](#authentication)`
- `[Core API](#core-api)`

## Authentication

### 1. Register a new user
```bash
curl -X POST https://mira-0gn4.onrender.com/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123",
    "email": "test@example.com",
    "role": "user"
  }'
```

### 2. Login and get a token
```bash
curl -X POST https://mira-0gn4.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123"
  }'
```
Save the token from the response:
```
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 3. Request a password reset
```bash
curl -X POST https://mira-0gn4.onrender.com/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com"
  }'
```
If DEBUG is enabled, you'll get a debug_token in the response.

### 4. Reset password with token
```bash
curl -X POST https://mira-0gn4.onrender.com/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{
    "token": "YOUR_RESET_TOKEN",
    "password": "newpassword123"
  }'
```

## Core API

### 1. Health Check (No Auth Required)
```bash
curl https://mira-0gn4.onrender.com/health
```

### 2. Login to Get Auth Token
```bash
curl -X POST https://mira-0gn4.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123"
  }'
```

Save the token from the response:
```
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 3. Chat API (Non-streaming)
```bash
curl -X POST https://mira-0gn4.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "stream": false
  }'
```

### 4. Chat API (Streaming)
```bash
curl -X POST https://mira-0gn4.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "Write a short poem about AI",
    "stream": true
  }'
```

### 5. Flow API
```bash
curl -X POST https://mira-0gn4.onrender.com/api/flow \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "Create a plan to build a simple website",
    "flow_type": "PLANNING"
  }'
```

### 6. Tools API
```bash
curl -X GET https://mira-0gn4.onrender.com/api/tools \
  -H "Authorization: Bearer $TOKEN"
```

### 7. Upload File API
```bash
curl -X POST https://mira-0gn4.onrender.com/api/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/file.txt"
```

### 8. List Files API
```bash
curl -X GET https://mira-0gn4.onrender.com/api/files \
  -H "Authorization: Bearer $TOKEN"
```

### 9. Chat API with Session ID (for continuing conversations)
```bash
# First request to get a session ID
RESPONSE=$(curl -X POST https://mira-0gn4.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "prompt": "Tell me about machine learning",
    "stream": false
  }')

# Extract session ID
SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')

# Second request using the same session ID
curl -X POST https://mira-0gn4.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"prompt\": \"Continue explaining about neural networks\",
    \"session_id\": \"$SESSION_ID\",
    \"stream\": false
  }"
```

## Integration Notes

The Agent Framework is configured to communicate with the RunPod AI service at `https://alhgtq3p5oelru-8888.proxy.runpod.net` using API key `7f8e9d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8`.

All API requests (except health check) require authentication with a valid JWT token obtained from the login endpoint.




FOR WINDOW CLI:


# Agent Framework API Documentation

This document provides comprehensive documentation for the Agent Framework API endpoints deployed at `https://mira-0gn4.onrender.com`.

## Windows Command Prompt Format

For Windows Command Prompt, use the following format for curl commands:

## Authentication

### 1. Register a new user
```cmd
curl -X POST https://mira-0gn4.onrender.com/auth/register -H "Content-Type: application/json" -d "{\"username\":\"testuser\",\"password\":\"securepassword123\",\"email\":\"test@example.com\",\"role\":\"user\"}"
```

### 2. Login and get a token
```cmd
curl -X POST https://mira-0gn4.onrender.com/auth/login -H "Content-Type: application/json" -d "{\"username\":\"testuser\",\"password\":\"securepassword123\"}"
```
Save the token from the response:
```cmd
set TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 3. Request a password reset
```cmd
curl -X POST https://mira-0gn4.onrender.com/auth/forgot-password -H "Content-Type: application/json" -d "{\"email\":\"test@example.com\"}"
```

### 4. Reset password with token
```cmd
curl -X POST https://mira-0gn4.onrender.com/auth/reset-password -H "Content-Type: application/json" -d "{\"token\":\"YOUR_RESET_TOKEN\",\"password\":\"newpassword123\"}"
```

## Core API

### 1. Health Check (No Auth Required)
```cmd
curl https://mira-0gn4.onrender.com/health
```

### 2. Chat API (Non-streaming)
```cmd
curl -X POST https://mira-0gn4.onrender.com/api/chat -H "Content-Type: application/json" -H "Authorization: Bearer %TOKEN%" -d "{\"prompt\":\"What is artificial intelligence?\",\"stream\":false}"
```

### 3. Chat API (Streaming)
```cmd
curl -X POST https://mira-0gn4.onrender.com/api/chat -H "Content-Type: application/json" -H "Authorization: Bearer %TOKEN%" -d "{\"prompt\":\"Write a short poem about AI\",\"stream\":true}"
```

### 4. Flow API
```cmd
curl -X POST https://mira-0gn4.onrender.com/api/flow -H "Content-Type: application/json" -H "Authorization: Bearer %TOKEN%" -d "{\"prompt\":\"Create a plan to build a simple website\",\"flow_type\":\"PLANNING\"}"
```

### 5. Tools API
```cmd
curl -X GET https://mira-0gn4.onrender.com/api/tools -H "Authorization: Bearer %TOKEN%"
```

### 6. Upload File API
```cmd
curl -X POST https://mira-0gn4.onrender.com/api/upload -H "Authorization: Bearer %TOKEN%" -F "file=@C:\path\to\your\file.txt"
```

### 7. List Files API
```cmd
curl -X GET https://mira-0gn4.onrender.com/api/files -H "Authorization: Bearer %TOKEN%"
```

### 8. Chat API with Session ID (for continuing conversations)
For Windows, this requires multiple commands:
```cmd
REM First request to get a session ID
curl -X POST https://mira-0gn4.onrender.com/api/chat -H "Content-Type: application/json" -H "Authorization: Bearer %TOKEN%" -d "{\"prompt\":\"Tell me about machine learning\",\"stream\":false}" > response.json

REM Extract session ID (requires jq or similar tool, or manual extraction)
REM If using jq:
for /f "tokens=*" %%a in ('type response.json ^| jq -r .session_id') do set SESSION_ID=%%a

REM Second request using the same session ID
curl -X POST https://mira-0gn4.onrender.com/api/chat -H "Content-Type: application/json" -H "Authorization: Bearer %TOKEN%" -d "{\"prompt\":\"Continue explaining about neural networks\",\"session_id\":\"%SESSION_ID%\",\"stream\":false}"
```

## Integration Notes

The Agent Framework is configured to communicate with the RunPod AI service at `https://alhgtq3p5oelru-8888.proxy.runpod.net` using API key `7f8e9d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9f8`.

All API requests (except health check) require authentication with a valid JWT token obtained from the login endpoint.
