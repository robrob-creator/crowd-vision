# CrowdVision - Garbage Detection Makefile
# ===========================================

.PHONY: help install run test clean lint format dev-setup check-deps

# Default target
help: ## Show this help message
	@echo "CrowdVision - Garbage Detection"
	@echo "==============================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Installation
install: ## Install Python dependencies
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install black flake8 pytest pytest-cov

# Virtual environment
venv: ## Create virtual environment
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Virtual environment created. Run 'source .venv/bin/activate' to activate."

venv-activate: ## Show command to activate virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"

source: ## Activate virtual environment (shows command)
	@echo "Run the following command to activate the virtual environment:"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Or use: make venv-activate"

# Running the application
run: ## Run the Streamlit application
	@echo "Starting CrowdVision application..."
	streamlit run app.py

run-dev: ## Run the application in development mode
	@echo "Starting CrowdVision in development mode..."
	streamlit run app.py --server.headless false --server.port 8501

# Testing
test: ## Run all tests
	@echo "Running tests..."
	python -m pytest test_deployment.py -v

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	python -m pytest test_deployment.py --cov=. --cov-report=html

# Code quality
lint: ## Run linting
	@echo "Running flake8 linting..."
	flake8 app.py detectron_detector.py test_deployment.py --max-line-length=100

format: ## Format code with black
	@echo "Formatting code with black..."
	black app.py detectron_detector.py test_deployment.py

check-format: ## Check code formatting
	@echo "Checking code formatting..."
	black --check app.py detectron_detector.py test_deployment.py

# Model testing
test-models: ## Test model loading
	@echo "Testing model loading..."
	python -c "
	from ultralytics import YOLO
	import os
	print('Testing YOLO models...')
	try:
	    # Test person detection model
	    model = YOLO('yolov8n.pt')
	    print('✅ Person detection model (yolov8n.pt) loaded successfully')
	except Exception as e:
	    print(f'❌ Failed to load yolov8n.pt: {e}')

	try:
	    # Test garbage detection model
	    if os.path.exists('models/garbage_detector.pt'):
	        model = YOLO('models/garbage_detector.pt')
	        print('✅ Garbage detection model loaded successfully')
	    else:
	        print('⚠️  Custom garbage model not found, using yolov8n.pt fallback')
	except Exception as e:
	    print(f'❌ Failed to load garbage model: {e}')
	"

# Firebase testing
test-firebase: ## Test Firebase connectivity
	@echo "Testing Firebase connectivity..."
	python -c "
	try:
	    import firebase_admin
	    from firebase_admin import credentials
	    import os

	    if os.path.exists('firebase_credentials.json'):
	        cred = credentials.Certificate('firebase_credentials.json')
	        firebase_admin.initialize_app(cred)
	        print('✅ Firebase initialized successfully')
	    else:
	        print('⚠️  Firebase credentials not found')
	except Exception as e:
	    print(f'❌ Firebase test failed: {e}')
	"

# Detection module testing
test-detection: ## Test detection module
	@echo "Testing detection module..."
	python -c "
	from detectron_detector import run_detection
	import multiprocessing
	import time

	# Test args
	test_args = {
	    'source': 'test.mp4',
	    'model': 'yolov8n.pt',
	    'garbage_model': 'models/garbage_detector.pt',
	    'location': 'Test Location',
	    'interval': 180,
	    'garbage_conf': 0.25,
	    'person_conf': 0.5,
	    'firebase_method': 'realtime',
	    'firebase_credentials': 'firebase_credentials.json',
	    'firebase_path': '/test',
	    'firebase_db_url': 'https://test.firebaseio.com',
	    'log_metrics': False,
	    'privacy': False
	}

	print('Testing detection module import and basic functionality...')
	try:
	    # Test import
	    from detectron_detector import run_detection
	    print('✅ Detection module imported successfully')

	    # Test with dummy source (will fail but should load models)
	    p = multiprocessing.Process(target=run_detection, args=(test_args,))
	    p.start()
	    time.sleep(2)  # Let it try to start
	    p.terminate()
	    p.join()
	    print('✅ Detection process started (expected to fail on missing video)')

	except Exception as e:
	    print(f'❌ Detection test failed: {e}')
	"

# Cleanup
clean: ## Clean up cache files and temporary files
	@echo "Cleaning up cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/

clean-models: ## Clean downloaded models (keeps custom models)
	@echo "Cleaning downloaded models..."
	rm -f yolov8n.pt
	rm -f yolov8s.pt
	rm -f yolov8m.pt
	rm -f yolov8l.pt
	rm -f yolov8x.pt

clean-all: clean clean-models ## Clean everything
	@echo "Cleaning everything..."
	rm -rf .venv/

# Dependencies check
check-deps: ## Check if all dependencies are installed
	@echo "Checking dependencies..."
	python -c "
	import sys
	deps = ['streamlit', 'cv2', 'numpy', 'firebase_admin', 'ultralytics']
	missing = []
	for dep in deps:
	    try:
	        __import__(dep)
	        print(f'✅ {dep}')
	    except ImportError:
	        missing.append(dep)
	        print(f'❌ {dep}')

	if missing:
	    print(f'\nMissing dependencies: {missing}')
	    print('Run: pip install -r requirements.txt')
	else:
	    print('\nAll dependencies installed!')
	"

# Development setup
dev-setup: venv install-dev ## Set up development environment
	@echo "Development environment setup complete!"
	@echo "Run: source .venv/bin/activate"
	@echo "Then: make run"

# Docker (if needed in future)
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t crowdvision .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8501:8501 crowdvision

# Deployment helpers
package-models: ## Package models for deployment
	@echo "Packaging models..."
	if [ -d "models" ]; then \
		tar -czf models.tar.gz models/; \
		echo "Models packaged as models.tar.gz"; \
	else \
		echo "No models directory found"; \
	fi

# Quick start
quick-start: check-deps test-models run ## Quick start the application
	@echo "Quick start complete!"

# Info
info: ## Show project information
	@echo "CrowdVision - Garbage Detection System"
	@echo "======================================"
	@echo "Python Version: $(shell python --version)"
	@echo "Streamlit Version: $(shell python -c "import streamlit; print(streamlit.__version__)" 2>/dev/null || echo "Not installed")"
	@echo "YOLO Version: $(shell python -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null || echo "Not installed")"
	@echo "Models Directory: $(shell ls -la models/ 2>/dev/null | wc -l) files"
	@echo "Firebase Config: $(shell ls firebase_credentials.json 2>/dev/null && echo "Found" || echo "Not found")"