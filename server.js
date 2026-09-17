// server.js - OpenAI → NVIDIA NIM Proxy (Clean Version)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ========== MIDDLEWARE ==========
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Log every request (helps debugging)
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

// ========== CONFIG ==========
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping (you can change these)
const MODEL_MAPPING = {
  'gpt-4': 'deepseek-ai/deepseek-v4-pro-0813',
  'gpt-4-turbo': 'z-ai/glm-5.3',
  'gpt-4o': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'claude-3-opus': 'google/gemma-4-31b-it',
  'claude-3-sonnet': 'z-ai/glm-5.3-flash',
  'gemini-pro': 'moonshotai/kimi-k3',
  // Add more if you want
};

// ========== ROUTES ==========

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    time: new Date().toISOString()
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Main chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'NIM_API_KEY is not set in environment variables',
          type: 'server_error',
          code: 500
        }
      });
    }

    const { model, messages, temperature, max_tokens, stream = false } = req.body;

    const nimModel = MODEL_MAPPING[model] || model;

    const payload = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 4096,
      stream
    };

    console.log(`→ Forwarding to NVIDIA model: ${nimModel}`);

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      payload,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000
      }
    );

    // ===== Streaming =====
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.pipe(res);
      return;
    }

    // ===== Non-streaming =====
    const data = response.data;

    const openaiResponse = {
      id: data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: data.created || Math.floor(Date.now() / 1000),
      model: model,
      choices: data.choices.map(choice => ({
        index: choice.index,
        message: {
          role: choice.message.role,
          content: choice.message.content || ''
        },
        finish_reason: choice.finish_reason
      })),
      usage: data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);

  } catch (error) {
    console.error('Proxy Error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message
    });

    const status = error.response?.status || 500;
    const message =
      error.response?.data?.message ||
      error.response?.data?.detail ||
      error.message ||
      'Internal server error';

    res.status(status).json({
      error: {
        message,
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.originalUrl} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ========== START ==========
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Models: http://localhost:${PORT}/v1/models`);
});
