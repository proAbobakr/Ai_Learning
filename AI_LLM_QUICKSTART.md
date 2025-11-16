# AI & LLM Course - Quick Start Guide

## 🚀 Get Started in 30 Minutes

This guide will get you up and running with the AI & LLM course in just 30 minutes.

---

## Step 1: Check Prerequisites (5 minutes)

### Required
- ✅ Python 3.10 or higher
- ✅ 200GB+ free disk space
- ✅ Text editor or IDE (VS Code recommended)
- ✅ Git installed

### Recommended
- ⭐ GPU with 8GB+ VRAM (or cloud access)
- ⭐ 16GB+ RAM
- ⭐ Basic Python knowledge

### Verify Python Installation
```bash
python --version
# Should show Python 3.10 or higher

python -c "import sys; print(f'Python {sys.version}')"
```

---

## Step 2: Environment Setup (10 minutes)

### Clone Repository (if not already done)
```bash
git clone <your-repo-url>
cd Ai_Learning
```

### Create Virtual Environment
```bash
# Create virtual environment
python -m venv ai_llm_env

# Activate it
# On Linux/Mac:
source ai_llm_env/bin/activate

# On Windows:
ai_llm_env\Scripts\activate
```

### Install Dependencies

**Option 1: Start with basics (fastest)**
```bash
# Install only Module 1 requirements
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

**Option 2: Install everything (takes 15-30 minutes)**
```bash
# Install all dependencies
pip install -r ai_llm_requirements.txt
```

**Option 3: Install with PyTorch GPU support**
```bash
# First, install PyTorch with CUDA support
# Visit https://pytorch.org to get the right command for your system

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r ai_llm_requirements.txt
```

### Verify Installation
```bash
python -c "import numpy as np; import pandas as pd; print('✓ NumPy and Pandas installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed'); print(f'  CUDA available: {torch.cuda.is_available()}')"
```

---

## Step 3: Get API Keys (5 minutes)

You'll need these for later modules:

### OpenAI (for LLM modules)
1. Go to https://platform.openai.com/api-keys
2. Create account and generate API key
3. Copy the key

### Create .env file
```bash
# Create .env file in the root directory
cat > .env << EOF
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
EOF
```

**Note**: You can get these later when you reach Module 5 (LLMs).

---

## Step 4: Launch Jupyter (5 minutes)

### Start Jupyter Lab
```bash
# Navigate to course directory
cd AI_LLM_Course

# Launch Jupyter
jupyter lab
```

This will open your browser with Jupyter Lab.

### Alternative: VS Code
If you prefer VS Code:
1. Install "Jupyter" extension
2. Open any .ipynb or .py file
3. Click "Run" to execute code cells

---

## Step 5: Run Your First Example (5 minutes)

### Test Python Skills
```bash
cd Module_1_Foundations/Week_1_Python/examples
python 01_python_essentials.py
```

You should see output demonstrating Python concepts!

### Test NumPy
```bash
python 02_numpy_fundamentals.py
```

You should see extensive NumPy examples running!

### Create Your First Jupyter Notebook
```python
# In Jupyter, create new notebook and run:

import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.title('My First AI Course Plot!')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.grid(True)
plt.show()

print("✓ Setup complete! Ready to learn AI!")
```

---

## Choose Your Learning Path

### 🔥 Intensive (6 months, 40-50 hours/week)
**Start here**: [Module 1, Week 1](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

**Your schedule**:
- Monday-Friday: 8 hours/day of focused learning
- Weekend: Review and projects
- Goal: Job-ready in 6 months

**Best for**: Career changers, bootcamp students, full-time learners

---

### 💼 Part-Time (12 months, 20-25 hours/week)
**Start here**: [Module 1, Week 1](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

**Your schedule**:
- Weekday mornings: 1 hour theory before work
- Weekday evenings: 2 hours coding after work
- Weekends: 5 hours each day for deep work
- Goal: Job-ready in 12 months

**Best for**: Working professionals

---

### 🎯 Weekend Warrior (18 months, 12-15 hours/week)
**Start here**: [Module 1, Week 1](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

**Your schedule**:
- Weekdays: 1 hour/day (theory reading)
- Saturdays: 6 hours (new concepts + coding)
- Sundays: 6 hours (projects + review)
- Goal: Job-ready in 18 months

**Best for**: Busy professionals, students

---

### 🧘 Self-Paced (24 months, 8-10 hours/week)
**Start here**: [Module 1, Week 1](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

**Your schedule**:
- 2 hours/day whenever you have time
- Deep learning at your own pace
- Goal: Mastery in 2 years

**Best for**: Casual learners, explorers

---

## Next Steps

### Immediate (Today)
1. ✅ Complete setup ← You are here!
2. 📖 Read [Main Course README](./AI_LLM_COURSE_README.md)
3. 📅 Review [Complete Timeline](./AI_LLM_COURSE_TIMELINE.md)
4. 🎯 Choose your learning path above
5. 🚀 Start [Week 1: Python for AI](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

### This Week
1. Complete Week 1 (Python & NumPy)
2. Run all 50+ code examples
3. Build mini-projects
4. Join community (Discord/Slack)

### This Month
1. Finish Module 1 (Foundations)
2. Build math intuition
3. Complete capstone project
4. Self-assess: all skills 4+/5

---

## Learning Resources

### Official Course Materials
- 📖 [Complete Course README](./AI_LLM_COURSE_README.md) - Full curriculum
- 📅 [Detailed Timeline](./AI_LLM_COURSE_TIMELINE.md) - Day-by-day breakdown
- 💻 [Code Examples](./AI_LLM_Course/) - 800+ examples
- 📝 [Requirements](./ai_llm_requirements.txt) - All dependencies

### Community & Support
- 💬 Discord: [Join here] (link in main README)
- 🐙 GitHub Issues: Report problems
- 📧 Email: [support email]
- 🌐 Forum: [link to forum]

### External Resources
- **Videos**: Fast.ai, 3Blue1Brown, Andrej Karpathy
- **Books**: Deep Learning Book, Speech & Language Processing
- **Papers**: ArXiv, Papers With Code
- **Practice**: Kaggle, HuggingFace

---

## Tips for Success

### 1. Code Every Day
```
✅ 30 minutes/day > 3.5 hours once a week
✅ Consistency is key
✅ Make it a habit
```

### 2. Build Projects
```
✅ Don't just read, implement
✅ Create portfolio from day 1
✅ Share your work publicly
```

### 3. Join Community
```
✅ Ask questions
✅ Help others
✅ Share progress
✅ Stay motivated
```

### 4. Track Progress
```
✅ Use todo lists
✅ Journal your learning
✅ Celebrate small wins
✅ Review regularly
```

### 5. Stay Curious
```
✅ Experiment beyond examples
✅ Read research papers
✅ Try new techniques
✅ Break things and fix them
```

---

## Common Issues & Solutions

### Issue: "pip install fails"
**Solution**:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try installing again
pip install -r ai_llm_requirements.txt
```

### Issue: "CUDA not available"
**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with matching CUDA version
# Visit: https://pytorch.org
```

### Issue: "Out of memory"
**Solution**:
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Use cloud GPUs (Colab, Kaggle)

### Issue: "Module not found"
**Solution**:
```bash
# Make sure virtual environment is activated
source ai_llm_env/bin/activate

# Install missing module
pip install <module_name>
```

---

## Quick Reference Commands

### Activate Environment
```bash
# Linux/Mac
source ai_llm_env/bin/activate

# Windows
ai_llm_env\Scripts\activate
```

### Start Jupyter
```bash
jupyter lab
# or
jupyter notebook
```

### Run Python Script
```bash
python script_name.py
```

### Install New Package
```bash
pip install package_name
```

### Check Python/Package Version
```bash
python --version
pip show package_name
```

### Deactivate Environment
```bash
deactivate
```

---

## Your First Week Roadmap

### Day 1: Setup & Python Basics (6-8 hours)
- ✅ Complete this quick start guide
- ✅ Run first examples
- ✅ Python essentials review

### Day 2: NumPy Fundamentals (6-8 hours)
- ✅ Array operations
- ✅ 30+ NumPy examples
- ✅ Practice exercises

### Day 3: Pandas & Data (6-8 hours)
- ✅ DataFrames
- ✅ Data cleaning
- ✅ Real datasets

### Day 4: Visualization (6-8 hours)
- ✅ Matplotlib
- ✅ Seaborn
- ✅ 10+ plot types

### Day 5: Integration (6-8 hours)
- ✅ Build data pipeline
- ✅ EDA tool
- ✅ Mini-projects

### Day 6-7: Review & Practice (8-10 hours)
- ✅ Consolidate learning
- ✅ Extra practice
- ✅ Prepare for Week 2

---

## Milestone: First Week Complete! 🎉

After Week 1, you should be able to:
- ✅ Write efficient Python for AI
- ✅ Manipulate NumPy arrays fluently
- ✅ Process data with Pandas
- ✅ Create professional visualizations
- ✅ Build complete data pipelines

**Confidence check**: All skills 3+/5? ✅ Move to Week 2!

---

## What's Next?

### Week 2: Linear Algebra
Master the mathematics powering AI:
- Vectors and matrices
- Eigenvalues and eigenvectors
- Applications in neural networks

**Start**: [Week 2: Linear Algebra](./AI_LLM_Course/Module_1_Foundations/Week_2_LinearAlgebra/README.md)

---

## Need Help?

### Stuck on something?
1. Re-read the section slowly
2. Run examples step-by-step
3. Check Stack Overflow
4. Ask in Discord/forum
5. Review external resources

### Want to go faster?
1. Skip optional sections
2. Use pre-built solutions
3. Focus on core concepts
4. Come back to details later

### Want to go deeper?
1. Read research papers
2. Implement variations
3. Build extra projects
4. Contribute to open source

---

## Motivation

> "The journey of a thousand miles begins with a single step."
> - Lao Tzu

You've taken the first step by setting up. Now keep going!

**Remember**:
- 🎯 Everyone starts as a beginner
- 📈 Progress compounds over time
- 🤝 Community support is available
- 💪 You can do this!

---

## Quick Stats

**After completing this course, you'll have**:
- ✅ 800+ code examples in your arsenal
- ✅ 25+ projects in your portfolio
- ✅ 1000+ hours of deliberate practice
- ✅ Job-ready AI/LLM engineering skills
- ✅ Deep understanding of modern AI

**Estimated time to job-ready**:
- Intensive: 6 months
- Part-time: 12 months
- Weekend: 18 months
- Self-paced: 24 months

---

**🚀 Ready? Let's start learning!**

**Next Step**: → [Begin Week 1: Python for AI](./AI_LLM_Course/Module_1_Foundations/Week_1_Python/README.md)

---

*Setup time: 30 minutes*
*First example: 5 minutes*
*First week: 40-50 hours*
*Job ready: 6-24 months*

**You've got this! 💪**
