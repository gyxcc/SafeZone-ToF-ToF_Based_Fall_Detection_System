# GitHub Push Instructions

This guide explains how to push this project to a **different GitHub account** (not your current Copilot-linked account).

## Option 1: Using GitHub CLI (Recommended)

### Step 1: Install Git (if not installed)
Download from: https://git-scm.com/download/win

### Step 2: Create Repository on GitHub
1. Log into your **other GitHub account** in browser
2. Click "+" → "New repository"
3. Name it: `tof-fall-detection`
4. **DO NOT** initialize with README (we already have one)
5. Click "Create repository"
6. Copy the repository URL: `https://github.com/OTHER_ACCOUNT/tof-fall-detection.git`

### Step 3: Open Terminal in Project Folder

```powershell
cd C:\Users\LTTC\Desktop\fyp\tof_fall_detection_release
```

### Step 4: Initialize Git and Configure for Other Account

```powershell
# Initialize git
git init

# Configure for your OTHER account (local config, won't affect other repos)
git config user.name "Your Other Account Username"
git config user.email "other-account-email@example.com"

# Verify configuration
git config user.name
git config user.email
```

### Step 5: Add Files and Commit

```powershell
# Add all files
git add .

# Check what will be committed (should NOT include .pt, .joblib, etc.)
git status

# Commit
git commit -m "Initial commit: ToF-based fall detection system"
```

### Step 6: Add Remote and Push

```powershell
# Add remote (use YOUR other account's repo URL)
git remote add origin https://github.com/OTHER_ACCOUNT/tof-fall-detection.git

# Push to main branch
git push -u origin main
```

### Step 7: Authenticate
When prompted, enter credentials for your **OTHER account**:
- Use Personal Access Token (PAT) instead of password
- Generate PAT at: GitHub → Settings → Developer settings → Personal access tokens

---

## Option 2: Using Different Git Credential

If you have issues with cached credentials:

### Clear Cached Credentials
```powershell
# Open Credential Manager
control /name Microsoft.CredentialManager

# Or via command
cmdkey /delete:git:https://github.com
```

### Use Credential Store per Repo
```powershell
# In the project folder, use this before push:
git config credential.helper store

# This will prompt for new credentials on next push
```

---

## Option 3: Using SSH Key (Most Reliable)

### Step 1: Generate SSH Key for Other Account
```powershell
ssh-keygen -t ed25519 -C "other-account-email@example.com" -f ~/.ssh/id_ed25519_other
```

### Step 2: Add SSH Key to Other GitHub Account
1. Copy key: `cat ~/.ssh/id_ed25519_other.pub`
2. GitHub → Settings → SSH Keys → New SSH Key
3. Paste and save

### Step 3: Configure SSH for Multiple Accounts
Create/edit `~/.ssh/config`:
```
Host github-other
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_other
```

### Step 4: Use SSH URL
```powershell
git remote add origin git@github-other:OTHER_ACCOUNT/tof-fall-detection.git
git push -u origin main
```

---

## Troubleshooting

### "Permission denied" or "Authentication failed"
- Clear credential cache (see Option 2)
- Use Personal Access Token instead of password
- Try SSH method (Option 3)

### "Repository not found"
- Check URL spelling
- Make sure repo is created on GitHub
- Verify you have write access

### "Refusing to merge unrelated histories"
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Large File Errors
GitHub has 100MB file limit. If you need to include model weights:

```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.joblib"

# Add LFS config
git add .gitattributes
git commit -m "Add LFS tracking"

# Then add and push
git add models/
git commit -m "Add model weights"
git push
```

---

## After Push: Upload Model Weights

Since model files are excluded from git:

1. **Google Drive / Dropbox**
   - Upload `best.pt` and `fall_classifier_v6.joblib`
   - Get shareable links
   - Update README.md with download links

2. **GitHub Releases**
   - Go to repo → Releases → Create new release
   - Attach model files as release assets
   - Tag version (e.g., v1.0.0)

3. **Hugging Face Hub** (for ML models)
   - Create model repo on huggingface.co
   - Upload weights there
   - Link in README

---

## Quick Reference

```powershell
# One-liner setup (run in project folder)
cd C:\Users\LTTC\Desktop\fyp\tof_fall_detection_release

git init
git config user.name "OTHER_USERNAME"
git config user.email "other@email.com"
git add .
git commit -m "Initial commit: ToF fall detection system"
git remote add origin https://github.com/OTHER_ACCOUNT/tof-fall-detection.git
git push -u origin main
```

Replace:
- `OTHER_USERNAME` with your other GitHub username
- `other@email.com` with that account's email
- `OTHER_ACCOUNT` with the account name in the URL
