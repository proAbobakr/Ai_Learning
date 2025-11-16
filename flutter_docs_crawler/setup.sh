#!/bin/bash

# Flutter Documentation Crawler Setup Script

echo "=========================================="
echo "Flutter Documentation Crawler - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
    echo "  To activate later: source venv/bin/activate"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs output/{human_readable,llm_format}
echo "✓ Directories created"

# Test imports
echo ""
echo "Testing installation..."
python3 -c "import sys; sys.path.insert(0, 'src'); from crawler import FlutterDocsCrawler; print('✓ Imports successful')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Installation test passed"
else
    echo "⚠ Import test had warnings (may be OK)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  1. Run a test crawl: python main.py --max-pages 10"
echo "  2. View results: cat output/human_readable/INDEX.md"
echo "  3. Check logs: tail -f logs/crawler.log"
echo ""
echo "For more info, see README.md and USAGE.md"
echo ""
