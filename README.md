# titanic - Machine Learning from Disaster

- This is my attempt at a Kaggle competition https://www.kaggle.com/c/titanic where I submit a csv containing list of survivors given a test list.
- The titanic.py code is run from my Google colab notebook titanic.ipynb here https://colab.research.google.com/drive/1hx7gT1fwjKl8R3EfxxHFFW_5YGPPxRZT?usp=sharing
- The custom function titanic_model() uses compile() method with loss function BinaryCrossentropy() and optimizer function Adam()
- Dense layers were at 64 using 'sigmoid' activation function.
- 100 epochs were used when training
- The train.csv and test.csv datasets were downloaded from Kaggle but I had to make changes to it for the code to work:
  - replacing spaces with "UNKNOWN" for Cabin column;
  - replacing spaces with "999" in columns Age and Fare;
  - replacing spaces with "X" in Embark column;
- I skipped calling evaluate() because I didn't know what to assign to the y_test label. Do I add another column called "Survived" in test.csv? What values would I use?
- Calling predict() method generates a prediction list of all 0 which is incorrect I think.

- Submission of model against test.csv dataset resulted in a submission.csv with 184 rows (survivors) out of 418 passengers.
- A 0.7416 accuracy.
- Screenshot
https://github.com/chromilo/titanic/blob/main/submission.JPG

- Leaderboard
https://github.com/chromilo/titanic/blob/main/leaderboard.JPG
