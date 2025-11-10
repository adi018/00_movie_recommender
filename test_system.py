#!/usr/bin/env python
"""
Integration test script to verify entire recommender system works end-to-end.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_data():
    """Check if preprocessed data exists."""
    data_file = PROJECT_ROOT / "data" / "books_processed.csv"
    if data_file.exists():
        print("✅ Data: Preprocessed CSV found")
        return True
    print("❌ Data: Preprocessed CSV NOT found")
    return False

def check_model():
    """Check if trained model exists."""
    model_file = PROJECT_ROOT / "models" / "recommender_model.pkl"
    embeddings_file = PROJECT_ROOT / "models" / "embeddings.npy"
    metadata_file = PROJECT_ROOT / "models" / "metadata.json"
    
    all_exist = (model_file.exists() and embeddings_file.exists() and metadata_file.exists())
    
    if all_exist:
        print("✅ Model: All artifacts found")
        print(f"   - Model: {model_file}")
        print(f"   - Embeddings: {embeddings_file}")
        print(f"   - Metadata: {metadata_file}")
        return True
    
    print("❌ Model: Some artifacts missing")
    if not model_file.exists():
        print(f"   - Missing: {model_file}")
    if not embeddings_file.exists():
        print(f"   - Missing: {embeddings_file}")
    if not metadata_file.exists():
        print(f"   - Missing: {metadata_file}")
    return False

def test_inference():
    """Test inference engine."""
    try:
        from recommender.inference import RecommenderEngine
        
        print("\n🧪 Testing Inference Engine...")
        engine = RecommenderEngine()
        
        test_query = "Science fiction with robots and AI"
        results = engine.recommend(test_query, k=3)
        
        if results and len(results) > 0:
            print(f"✅ Inference: Successfully got {len(results)} recommendations")
            for i, rec in enumerate(results[:1], 1):
                print(f"\n   Top Result: {rec['title']}")
                print(f"   By: {rec['authors']}")
                print(f"   Similarity: {rec['similarity_score']:.1%}")
            return True
        else:
            print("❌ Inference: No recommendations returned")
            return False
            
    except Exception as e:
        print(f"❌ Inference: Error - {str(e)}")
        return False

def test_imports():
    """Test if all modules can be imported."""
    try:
        print("\n📦 Checking Module Imports...")
        from recommender import preprocess, embeddings, train, inference
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("🧪 LLM BOOK RECOMMENDER SYSTEM - INTEGRATION TEST")
    print("=" * 70)
    
    results = []
    
    # Test 1: Check imports
    results.append(("Module Imports", test_imports()))
    
    # Test 2: Check data
    results.append(("Preprocessed Data", check_data()))
    
    # Test 3: Check model artifacts
    results.append(("Model Artifacts", check_model()))
    
    # Test 4: Test inference
    results.append(("Inference Engine", test_inference()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 70)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Run Streamlit app: streamlit run app.py")
        print("   2. Or deploy with Docker: docker-compose up --build")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
