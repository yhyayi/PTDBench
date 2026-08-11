export OPENCODE_CONFIG_CONTENT='{
  "provider": {
    "sglang": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "SGLang",
      "options": {
        "baseURL": "http://127.0.0.1:30004/v1",
        "apiKey": "EMPTY"
      },
      "models": {
        "/model": {
          "name": "Qwen3.5-9B"
        }
      }
    }
  }
}'

/home/test/test1706/.nvm/versions/node/v24.15.0/bin/opencode \
  /home/test/test1706/yuanjiarui/data_call \
  --pure \
  -m 'sglang//model'




/home/test/test1706/.nvm/versions/node/v24.15.0/bin/opencode run \
  --pure \
  --dir /home/test/test1706/yuanjiarui/data_call \
  -m 'sglang//model' \
  "检查当前项目并告诉我它的主要目录结构"