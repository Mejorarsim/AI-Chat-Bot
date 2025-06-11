#!/usr/bin/env python3
"""
COMPLETE DEPENDENCY INSTALLER FOR CHATBOT PROJECT
=================================================
Run this file FIRST to install all required dependencies for the chatbot.
Compatible with Spyder IDE and all Python environments.

Instructions:
1. Save this file as 'install_dependencies.py'
2. Run it in Spyder by pressing F5
3. Wait for all installations to complete
4. Then run the main chatbot file

Author: AI Assistant
Date: 2025
"""

import subprocess
import sys
import os
import importlib

print("🚀 CHATBOT DEPENDENCY INSTALLER")
print("=" * 50)
print("This will install ALL required packages for the chatbot project.")
print("Please wait while dependencies are installed...")
print("=" * 50)

def install_package(package_name, import_name=None):
    """Install a package using pip and verify installation"""
    if import_name is None:
        import_name = package_name.split('>=')[0].split('==')[0]
    
    try:
        # Try to import first to see if already installed
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Already installed")
        return True
    except ImportError:
        try:
            print(f"📦 Installing {package_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, 
                "--upgrade", "--quiet"
            ])
            
            # Verify installation
            importlib.import_module(import_name)
            print(f"✅ {package_name} - Successfully installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False
        except ImportError:
            print(f"⚠️  {package_name} installed but import failed")
            return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    try:
        print("🔧 Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"
        ])
        print("✅ Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Could not upgrade pip, continuing anyway...")
        return False

def install_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("📚 Downloading NLTK data...")
        
        # Download essential NLTK data
        datasets = ['punkt', 'wordnet', 'omw-1.4', 'stopwords', 'vader_lexicon']
        
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
                print(f"✅ NLTK {dataset} downloaded")
            except Exception as e:
                print(f"⚠️  Could not download {dataset}: {e}")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data download failed: {e}")
        return False

def test_installations():
    """Test that all critical packages work"""
    print("\n🧪 Testing installations...")
    
    tests = [
        ("TensorFlow", "tensorflow", "import tensorflow as tf; tf.__version__"),
        ("NLTK", "nltk", "import nltk; nltk.__version__"),
        ("NumPy", "numpy", "import numpy as np; np.__version__"),
        ("Matplotlib", "matplotlib", "import matplotlib; matplotlib.__version__"),
        ("Flask", "flask", "import flask; flask.__version__"),
        ("Pickle", "pickle", "import pickle"),
        ("JSON", "json", "import json"),
        ("Random", "random", "import random"),
        ("Tkinter", "tkinter", "import tkinter as tk")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, module_name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {test_name}: Working")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: Failed - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} packages working")
    return passed == total

def main():
    """Main installation process"""
    
    # Step 1: Upgrade pip
    print("\n1️⃣ UPGRADING PIP")
    print("-" * 30)
    upgrade_pip()
    
    # Step 2: Install core packages
    print("\n2️⃣ INSTALLING CORE PACKAGES")
    print("-" * 30)
    
    # List of all required packages
    packages = [
        # Core ML/AI packages
        ("tensorflow>=2.13.0", "tensorflow"),
        ("numpy>=1.24.0", "numpy"),
        ("scikit-learn>=1.3.0", "sklearn"),
        
        # NLP packages
        ("nltk>=3.8.1", "nltk"),
        
        # Data visualization
        ("matplotlib>=3.7.0", "matplotlib"),
        ("seaborn>=0.12.0", "seaborn"),
        
        # Web framework
        ("flask>=2.3.0", "flask"),
        ("flask-cors>=4.0.0", "flask_cors"),
        
        # Development tools
        ("jupyter>=1.0.0", "jupyter"),
        ("ipykernel>=6.25.0", "ipykernel"),
        
        # Utilities
        ("pandas>=2.0.0", "pandas"),
        ("python-dotenv>=1.0.0", "dotenv"),
        ("requests>=2.31.0", "requests"),
    ]
    
    # Install each package
    success_count = 0
    for package, import_name in packages:
        if install_package(package, import_name):
            success_count += 1
    
    print(f"\n📦 Package Installation: {success_count}/{len(packages)} successful")
    
    # Step 3: Install NLTK data
    print("\n3️⃣ INSTALLING NLTK DATA")
    print("-" * 30)
    install_nltk_data()
    
    # Step 4: Test installations
    print("\n4️⃣ TESTING INSTALLATIONS")
    print("-" * 30)
    all_working = test_installations()
    
    # Step 5: Final summary
    print("\n" + "=" * 50)
    if all_working and success_count >= len(packages) - 2:  # Allow 2 optional packages to fail
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("✅ All dependencies are ready for the chatbot project")
        print("\n🚀 NEXT STEPS:")
        print("1. Close this script")
        print("2. Open 'complete_chatbot_spyder.py' in Spyder")
        print("3. Run the chatbot script (F5)")
        print("4. Enjoy your AI chatbot!")
        
        # Create a success marker file
        with open("installation_complete.txt", "w") as f:
            f.write("Dependencies installed successfully!\nDate: " + str(subprocess.check_output([sys.executable, "-c", "import datetime; print(datetime.datetime.now())"], text=True).strip()))
        
    else:
        print("⚠️  INSTALLATION COMPLETED WITH SOME ISSUES")
        print("Some packages may not have installed correctly.")
        print("Try running this script again or install manually:")
        print("pip install tensorflow nltk matplotlib flask numpy")
        
        # Show manual installation commands
        print("\n🔧 MANUAL INSTALLATION COMMANDS:")
        print("Copy and paste these commands in Anaconda Prompt or Command Prompt:")
        print("pip install --upgrade pip")
        for package, _ in packages:
            print(f"pip install {package}")
    
    print("=" * 50)
    input("\nPress Enter to close this installer...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Try running as administrator or check your internet connection")
    finally:
        print("\nInstaller finished.")