# Scientific Programming in Python – submission 3

### Project title: [title]

### Student name: [name]
  
This file should contain a refined version of your initial README, reflecting the final state of development. If it isn't obvious, you should provide a brief guide to how to run your code. Your code should be placed in `.py` or `.ipynb` files in this repo, together with some example data (if required) and example outputs (e.g., pdf plots).

As a reminder, your program should (as a rough guide)...
* be written as an executable module (.py file) or Jupyter notebook (.ipynb)
* do something meaningful: analyse real data or perform a simulation
* define at least two user functions (but typically more)
* make use of appropriate specialist modules
* produce at least one informative plot
* comprise at least 50 lines of actual code (excluding comments, imports and other 'boilerplate')
* contain no more than 1000 lines in total (if you have written more, please isolate an individual element)

Particular credit will be given for code that:
*  is easily readable and comprehensible,
*  is efficient (i.e. in terms of memory and CPU time),
*  is elegant (i.e. not unnecessarily lengthy or complex),
*  makes use of appropriate Python language elements,
*  makes use of appropriate Python modules,
*  is clear where aspects of the implementation are incomplete.

#### Copying over files from your sub2 repository

You can just copy files over from your sub2 repository into your sub3 repository and commit, then continue editing here. Alternatively, to merge your full sub2 commit history into this repository, you can do the following (substituting your username):
```sh
# clone your sub3 repo (you can use the https url here if you haven't set up ssh keys)
git clone git@github.com:mpags-python/coursework2021-sub3-bamford.git sub3

# move into that repo
cd sub3

# add the sub2 repository as another remote
git remote add sub2 git@github.com:mpags-python/coursework2021-sub2-bamford.git

# make sure the remote is known to git
git fetch sub2

# merge in your sub2 repository
# Note that this will overwrite the README.md and any other files in sub3 repo with the sub2 version.
# If you have already made changes in sub3 that you want to keep, you can leave off the -Xtheirs,
# but you will have to resolve the merge conflicts and commit manually
git merge sub2/main --allow-unrelated-histories -Xtheirs

# push changes to github
git push
```
