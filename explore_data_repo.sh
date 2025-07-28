# ============================================
# SIMPLE DORA EXPLORATION - NO TREE REQUIRED
# Copy and paste each section separately
# ============================================

# 1. NAVIGATE TO DORA REPO
cd ~/peft-scientific/DoRA
pwd
echo "✅ Current directory: $(pwd)"

# 2. SHOW TOP-LEVEL STRUCTURE
echo -e "\n📁 DoRA Repository Contents:"
ls -la

# 3. EXAMINE EACH MAIN DIRECTORY
echo -e "\n🔍 EXAMINING DIRECTORIES:"

if [ -d "commonsense_reasoning" ]; then
    echo -e "\n🧠 COMMONSENSE_REASONING:"
    ls -la commonsense_reasoning/
    echo "Python files:"
    find commonsense_reasoning/ -name "*.py" 2>/dev/null | head -5
fi

if [ -d "instruction_tuning_dvora" ]; then
    echo -e "\n🎯 INSTRUCTION_TUNING_DVORA:"
    ls -la instruction_tuning_dvora/
    echo "Python files:"
    find instruction_tuning_dvora/ -name "*.py" 2>/dev/null | head -5
fi

# Check for other instruction tuning directories
for dir in instruction_tuning*; do
    if [ -d "$dir" ] && [ "$dir" != "instruction_tuning_dvora" ]; then
        echo -e "\n🎯 $dir:"
        ls -la "$dir"/
    fi
done

if [ -d "visual_instruction_tuning" ]; then
    echo -e "\n🖼️ VISUAL_INSTRUCTION_TUNING:"
    ls -la visual_instruction_tuning/
    echo "Python files:"
    find visual_instruction_tuning/ -name "*.py" 2>/dev/null | head -5
fi

if [ -d "image_video_text_understanding" ]; then
    echo -e "\n🎬 IMAGE_VIDEO_TEXT_UNDERSTANDING:"
    ls -la image_video_text_understanding/
    echo "Python files:"
    find image_video_text_understanding/ -name "*.py" 2>/dev/null | head -5
fi

# 4. LOOK FOR EXECUTABLE SCRIPTS
echo -e "\n🚀 EXECUTABLE SCRIPTS:"
find . -name "*.py" -exec grep -l "if __name__ == '__main__'" {} \; 2>/dev/null | head -10

# 5. LOOK FOR REQUIREMENTS/SETUP FILES
echo -e "\n📦 SETUP FILES:"
find . -name "requirements*.txt" -o -name "setup.py" -o -name "environment*.yml" 2>/dev/null

# 6. CHECK FOR README FILES
echo -e "\n📖 README FILES:"
find . -name "README*" -o -name "readme*" 2>/dev/null

# 7. FIND CONFIGURATION FILES
echo -e "\n⚙️ CONFIGURATION FILES:"
find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "config*" 2>/dev/null | head -10

echo -e "\n✅ Repository exploration complete!"
