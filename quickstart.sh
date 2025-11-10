#!/bin/bash
# Quick Start Script for LLM Book Recommender System

echo "🚀 LLM Book Recommender - Quick Start"
echo "===================================="

# Check Python
echo "✓ Checking Python installation..."
python3 --version

# Step 1: Create virtual environment
echo ""
echo "📦 Step 1: Creating virtual environment..."
python3 -m venv .venv_llm_recommender
source venv/bin/activate

# Step 2: Install dependencies
echo "📥 Step 2: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Preprocess data
echo "🔄 Step 3: Preprocessing data..."
python -m recommender.preprocess

# Step 4: Train model
echo "🏋️  Step 4: Training model (this may take 1-2 minutes)..."
python -m recommender.train

# Step 5: Test inference
echo "🧪 Step 5: Testing inference..."
python -m recommender.inference

# Step 6: Run Streamlit app
echo "🎨 Step 6: Launching Streamlit app..."
echo "📍 App will open at: http://localhost:8501"
streamlit run app.py

echo ""
echo "✅ Setup complete!"
