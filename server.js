// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Reasoning display toggle
const SHOW_REASONING = false;

// Thinking mode toggle
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
  'gpt-4': 'minimaxai/minimax-m3',
  'gpt-4-turbo': 'minimaxai/minimax-m3',
  'gpt-4o': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'claude-3-opus': 'nvidia/nemotron-3-ultra',
  'claude-3-sonnet': 'minimaxai/minimax-m3',
  'gemini-pro': 'minimaxai/minimax-m3'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    // Model selection
    // Uses the mapped NVIDIA model if one exists.
    // Otherwise passes the supplied model directly to NVIDIA.
    const nimModel = MODEL_MAPPING[model] || model;

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE
        ? {
            chat_template_kwargs: {
              thinking: true
            }
          }
        : undefined,
      stream: stream || false
    };

    // Make request to NVIDIA NIM API
    // Automatically retries temporary 429 responses.
    let response;
    const maxRetries = 3;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            responseType: stream ? 'stream' : 'json'
          }
        );

        break;
      } catch (error) {
        const status = error.response?.status;

        // If it isn't a 429, or we've exhausted retries,
        // pass the error to the main error handler.
        if (status !== 429 || attempt === maxRetries) {
          throw error;
        }

        // Respect NVIDIA's Retry-After header if available.
        const retryAfter = Number(
          error.response?.headers?.['retry-after']
        );

        // Otherwise use exponential backoff:
        // 1 second, 2 seconds, 4 seconds
        const delay =
          Number.isFinite(retryAfter) && retryAfter > 0
            ? retryAfter * 1000
            : Math.pow(2, attempt) * 1000;

        console.log(
          `NVIDIA returned 429. Retrying in ${delay / 1000}s...`
        );

        await new Promise(resolve =>
          setTimeout(resolve, delay)
        );
      }
    }

    // Handle streaming response
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) {
            return;
          }

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const reasoning =
                data.choices[0].delta.reasoning_content;

              const content =
                data.choices[0].delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent =
                    '<think>\n' + reasoning;

                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent +=
                    '</think>\n\n' + content;

                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content =
                    combinedContent;

                  delete data.choices[0].delta.reasoning_content;
                }
              } else {
                if (content) {
                  data.choices[0].delta.content = content;
                } else {
                  data.choices[0].delta.content = '';
                }

                delete data.choices[0].delta.reasoning_content;
              }
            }

            res.write(
              `data: ${JSON.stringify(data)}\n\n`
            );
          } catch (e) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', err => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // Transform NIM response to OpenAI format
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent =
            choice.message?.content || '';

          if (
            SHOW_REASONING &&
            choice.message?.reasoning_content
          ) {
            fullContent =
              '<think>\n' +
              choice.message.reasoning_content +
              '\n</think>\n\n' +
              fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('NVIDIA/Proxy error:', {
      status: error.response?.status,
      data: error.response?.data,
      message: error.message
    });

    const status =
      error.response?.status || 500;

    res.status(status).json({
      error: {
        message:
          error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(
    `OpenAI to NVIDIA NIM Proxy running on port ${PORT}`
  );

  console.log(
    `Health check: http://localhost:${PORT}/health`
  );

  console.log(
    `Reasoning display: ${
      SHOW_REASONING ? 'ENABLED' : 'DISABLED'
    }`
  );

  console.log(
    `Thinking mode: ${
      ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'
    }`
  );
});
