# Football prediction

## Table of Contents
* [About](#about)
* [Getting Started](#getting-started)
  * [Requirements](#requirements)
  * [Installation](#installation)
* [Usage](#usage)
* [Author](#author)

## About
<details><summary>CLICK ME</summary>
Application Flask using Machine Learning model to predict the results of the L1's matches through the season.
The model is a Random Forest learning from previous matches.
The model attempts to predict the probability to win or to lose using its statistics (shoots, goals, ...) shifted and also those of the opponent, considering the strength of the team in the time, if the team is at home...

The data are provided by the site [football-data.co.uk](https://www.football-data.co.uk/) and are updated each week. Then the model re-trained each week.
The model is `team-focus` thus to predict a match result we do it for each of the both teams and we average the score to have the probability.

The app allows to display the probability of each results for the day given.
</details>

## Getting Started
###  Requirements
It requires the following to run:
- flask
- scikit-learn
- pandas
- numpy

### Installation
```
git clone https://github.com/mjclm/football-prediction.git
```

## Visual
The input is the season's day wanted:
![app_step1](/image/app_step1.png)
After clicking on `predict!`, we obtain the results:
We have for each match, the home team, the away team, the probability to win for the home team, the probability of drawback and the probability to win for the away team.
![app_step2](/image/app_step2.png)

## Usage
It is easy to use, just run the following:
```sh
python main.py
```

And open the app in your browser.

## Author
Mickael JUILLET
