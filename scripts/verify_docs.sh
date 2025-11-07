#!/bin/bash

# Resonance Neural Networks - Documentation Verification
# Genovo Technologies Research Team
# Lead: Oluwatosin Afolabi (afolabi@genovotech.com)

echo "=============================================="
echo "RESONANCE NEURAL NETWORKS - DOCUMENTATION CHECK"
echo "Genovo Technologies - Confidential"
echo "=============================================="
echo ""

echo "📁 Checking Documentation Structure..."
echo ""

# Check if docs directory exists
if [ -d "docs" ]; then
    echo "✅ docs/ directory exists"
else
    echo "❌ docs/ directory missing"
    exit 1
fi

# List all documentation files
echo ""
echo "📄 Documentation Files:"
echo ""

cd docs

files=(
    "README.md"
    "INDEX.md"
    "HEADER.md"
    "ARCHITECTURE.md"
    "GETTING_STARTED.md"
    "IMPLEMENTATION_STATUS.md"
    "V2_FEATURES.md"
    "IMPLEMENTATION_SUMMARY.md"
    "COMPLETE_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
    fi
done

cd ..

echo ""
echo "🔒 Checking Confidentiality Notices..."
echo ""

# Check root level files
if [ -f "CONFIDENTIAL.md" ]; then
    echo "  ✅ CONFIDENTIAL.md exists"
else
    echo "  ❌ CONFIDENTIAL.md missing"
fi

if [ -f "LICENSE" ]; then
    echo "  ✅ LICENSE updated"
    if grep -q "Genovo Technologies" LICENSE; then
        echo "  ✅ License contains Genovo Technologies attribution"
    else
        echo "  ⚠️  License may need updating"
    fi
else
    echo "  ❌ LICENSE missing"
fi

echo ""
echo "📧 Checking Contact Information..."
echo ""

# Check for email in README
if grep -q "afolabi@genovotech.com" README.md; then
    echo "  ✅ Contact email found in README.md"
else
    echo "  ⚠️  Contact email not found in README.md"
fi

# Check for Genovo Technologies attribution
if grep -q "Genovo Technologies" README.md; then
    echo "  ✅ Genovo Technologies attribution in README.md"
else
    echo "  ⚠️  Genovo Technologies attribution not found"
fi

echo ""
echo "=============================================="
echo "Documentation Structure Verification Complete"
echo "=============================================="
echo ""
echo "For questions, contact:"
echo "Oluwatosin Afolabi - afolabi@genovotech.com"
echo ""
