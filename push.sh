# 1. Extract the zip locally.
unzip alpha-blended-eaf.zip
cd alpha-blended-eaf

# 2. Initialize git and make the first commit.
git init
git add .
git commit -m "Initial commit: cross-validated EAF evaluation pipeline"

# 3. Create the repo on GitHub.com (public, no README/license/gitignore —
#    those are already in the zip).

# 4. Push.
git branch -M main
git remote add origin https://github.com/rommelskii/alpha-blended-eaf.git
git push -u origin main
