#!/bin/bash
# ORRRG Installation and Setup Script
# ===================================

set -e  # Exit on error

echo "ORRRG - Omnipotent Research and Reasoning Reactive Grid"
echo "======================================================="
echo "Self-Organizing Core Integration System Setup"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc -l) -eq 0 ]]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Check if we're in the right directory
if [[ ! -f "orrrg_main.py" ]] || [[ ! -d "core" ]]; then
    echo "Error: Please run this script from the ORRRG root directory"
    exit 1
fi

echo "✓ ORRRG root directory confirmed"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip
echo "✓ pip upgraded"

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Install ORRRG in development mode
echo "Installing ORRRG in development mode..."
pip install -e .
echo "✓ ORRRG installed"

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p temp
echo "✓ Directories created"

# Set up .gitignore for generated files
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# ORRRG specific
logs/
data/
temp/
*.log
orrrg.db

# Component-specific ignores
components/*/node_modules/
components/*/build/
components/*/dist/
components/*/.git/
EOF

echo "✓ .gitignore configured"

# Test the installation
echo ""
echo "Testing installation..."
python3 -c "from core import SelfOrganizingCore; print('✓ Core module imports successfully')"

echo ""
echo "Installation complete!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run ORRRG interactively: python3 orrrg_main.py --mode interactive"
echo "  3. Or run in daemon mode: python3 orrrg_main.py --mode daemon"
echo ""
echo "For help: python3 orrrg_main.py --help"
echo ""

# Display system status
echo "Discovered components:"
ls -1 components/ 2>/dev/null | head -8 | while read component; do
    echo "  - $component"
done

echo ""
echo "Ready to organize and reason across all domains! 🚀"