[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6268779&assignment_repo_type=AssignmentRepo)
# Scientific Programming in Python – submission 2

### Project title: [title]

### Student name: [name]
  
This file should contain a refined version of your initial README, reflecting the current state of development. Your code should be placed in `.py` or `.ipynb` files in this repo, possibly together with some example data and/or preliminary results.

For this submission, only a roughly working version of your code is expected. It is intended that this be incomplete and/or contain bugs. Even so, do try to make the code in this submission reasonably easy to understand. Ideally, document what still remains to be done, via comments in the code, this README and/or [GitHub issues](https://guides.github.com/features/issues/).

#### Copying over files from your sub1 repository

You can just copy files over from your sub1 repository into your sub2 repository and commit, then continue editing here. Alternatively, to merge your full sub1 commit history into this repository, you can do the following (substituting your username):
```sh
# clone your sub2 repo (you can use the https url here if you haven't set up ssh keys)
git clone git@github.com:mpags-python/coursework2021-sub2-bamford.git sub2

# move into that repo
cd sub2

# add the sub1 repository as another remote
git remote add sub1 git@github.com:mpags-python/coursework2021-sub1-bamford.git

# make sure the remote is known to git
git fetch sub1

# merge in your sub1 repository
# Note that this will overwrite the README.md and any other files in sub2 repo with the sub1 version.
# If you have already made changes in sub2 that you want to keep, you can leave off the -Xtheirs,
# but you will have to resolve the merge conflicts and commit manually
git merge sub1/main --allow-unrelated-histories -Xtheirs

# push changes to github
git push
```
