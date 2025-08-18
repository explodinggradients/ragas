# Ragas Webview CLI

A web-based file viewer with React frontend and Python backend. Easily spin up a web interface to browse and view local project files.

## Features

- 🚀 **Standalone CLI** - Single command to start web server
- ⚛️ **React Frontend** - Modern web interface for file browsing
- 🐍 **Python Backend** - FastAPI server with file serving capabilities
- 🔄 **Development Mode** - Hot reload for both frontend and backend
- 📦 **Production Ready** - Bundled React app served by Python server

## Installation

### Using uv (recommended)

```shell
  #For development (both React + Python servers):
  uv run python scripts/dev.py <directory>
  # uv run python scripts/dev.py ./logs

  #For production (Python server only):
  uv run python -m ragas_webview_cli <directory>
  # uv run python -m ragas_webview_cli ./logs

```


```bash
# Install with uv
uv init

# Or install from local directory
cd webview-poc
uv sync

uv run python scripts/dev.py

uv run python -m ragas_webview_cli logs
```

### Using pip

```bash
pip install ragas-webview-cli

# Or install from local directory
cd webview-poc
pip install -e .
```

## Usage

### Production Mode (Bundled)

```bash
# Start the webview server
uv run ragas-webview-cli --port 8000 --host 127.0.0.1

# Or use Python directly
uv run python -m ragas_webview_cli.cli --port 8000
```

### Development Mode

```bash
# Run both React dev server and Python API server
uv run python scripts/dev.py

# This starts:
# - React dev server on http://localhost:3000 (with hot reload)
# - Python API server on http://localhost:8000
```

### Building React Bundle

```bash
# Build React app for production
uv run python scripts/build.py

# Or build directly with npm
cd ragas-webview
npm run build:bundle
```

## Development Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- uv (recommended) or pip

### Setup

```bash
# Clone and setup with uv
git clone <repo>
cd webview-poc
uv sync

# Install React dependencies
cd ragas-webview
npm install
cd ..

# Run development environment
uv run python scripts/dev.py
```

## Project Structure

```
webview-poc/
├── README.md
├── pyproject.toml              # ← Package configuration
├── uv.lock                     # ← UV lock file
├── js-bundle/                  # ← Built React app (not packaged)
├── ragas-webview/              # ← React source (not packaged)
├── scripts/                    # ← Development scripts (not packaged)
│   ├── dev.py                  # ← Development server script
│   └── build.py                # ← Build script
└── ragas_webview_cli/          # ← Main package (packaged in pip)
    ├── __init__.py
    ├── cli.py                  # ← CLI entry point
    ├── config.py               # ← Configuration
    └── server.py               # ← FastAPI server
```

## Configuration

All server ports and hosts are configured in `ragas_webview_cli/config.py`:

```python
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_HOST = "127.0.0.1"
FRONTEND_HOST = "127.0.0.1"
```

## API Endpoints

- `GET /` - Serves React app (production) or API info (development)
- `GET /api/health` - Health check endpoint
- `GET /assets/*` - Static assets (CSS, JS, images)

## License

MIT License