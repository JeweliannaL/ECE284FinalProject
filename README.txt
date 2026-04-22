Uploading a README so we have at least one file on here to start.

You can just delete and upload your model as you go if you want or you can set up git (you can look up a tutorial to do this online) on your computer. This will make it where when you “push” aka add your changes to GitHub, it will track everything you added and deleted, which can be useful for version control and for the group to see what you are working on.

My experience with using GitHub is limited, but this is what I’ve always done and it has always worked for me:
1) Set up Git on your computer.

2) Assuming your model is already uploaded to the repo (or if you want to work on someone else’s model) go to the GitHub repo, click the “main” button in the top left corner, and create a new branch. You can name it something unique e.g. Jewels—03.30.2026. Doing this will update the current file on your desktop to the version on GitHub (and also presumably the most up to date). To be clear, the version on your computer will not automatically update if somebody makes changes to the model on GitHub, you NEED to make a new branch to see any changes they made or do “git pull” in the VS terminal on your current branch.

3) To make sure you are on the right branch (at least for vs code) it will have the name of your branch on the bottom left corner. To change to you branch, in the vs code terminal you can put: git checkout branchname (e.g git checkout Jewels—03.30.2026) If you don’t see your branch put git fetch before this

4) Once you have the most up to date model, you can make your changes. I typically like to push every time I make a large change, but that’s up to you. If you are working in VS code and are ready to upload you changes to GitHub, go to the terminal on the bottom and put this in order:
git add .
git commit -m “your message here”
git push

For your message, write a quick note on what you changed so the team can follow (e.g. made xyz changes to data preprocessing)

5) Once you have done git push, go back to GitHub, and check the pull requests tab. You then select create pull request and it will check to see if there are any issues with merging. If not, you can click commit. Your changes are now on GitHub!

There’s probably other ways to also do this but this is what has worked for me :)