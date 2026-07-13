# Model Bench

OpenAI Chat API load testing tool with an interactive curses-based UI.

Loads model endpoint configs from `~/.junie-local/models/*.json`, lets you pick an endpoint and tune parameters, then runs concurrent load tests and prints statistics.

Uses only Python stdlib (`urllib`, `curses`, `concurrent.futures`).

## Usage

```bash
python3 load_test.py
# or
./start.sh
```