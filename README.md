# Predicting Hotel Cancellations
Using machine learning to predict hotel reservation cancellations to maximize profit.

# About the data
This dataset was created as part of an open hotel booking dataset from [Antonio, Almeida and Nunes, 2019](https://www.sciencedirect.com/science/article/pii/S2352340918315191#f0010). I originally found it on the `tidytuesday` page on GitHub. A full description of the dataset can be found [here](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md). 

There was relatively little cleaning that I had to do for this dataset. I did, however, choose to conduct some feature engineering in order to create some new, more descriptive features. They are listed below:
- `total_nights`: this combined the `stays_in_weekend_nights` and `stays_in_week_nights` features.
- `party_size`: this combined the `adults`, `children`, and `babies` columns.
-  `is_family`: a boolean column I created using `adults`, `children`, and `babies`.
- `room_difference`: a boolea column combining `reserved_room_type` and `assigned_room_type` indicating if there was a difference between the guest's reserved and assigned rooms.

# EDA
I conducted some exploratory data analysis to try and determine which features contain the most signal in the data. These features will be the strongest indicators of whether or not a cancellation will occur.

<p align="center">
  <img width="900" height="450" src="img/is_repeat_guest.png">
</p>

There is a lot of signal in this data, which is to be expected given that the split is occurring on prior stays at the same hotel. Unfortunately, because the data is so imbalanced this feature may not be the best one to use, since the vast majority of the data will fall into one category. I will reserve it for now and perhaps add it in later on.

<p align="center">
  <img width="900" height="450" src="img/hotels.png">
</p>

<p align="center">
  <img width="900" height="450" src="img/num_special_requests.png">
</p>

<p align="center">
  <img width="900" height="450" src="img/total_nights.png">
</p>

These graphs all feature quite a bit of signal in the data, and these features are more balanced than the `is_repeat_guest` feature. For my baseline model, I'll use these features and a few others.

<p align="center">
  <img width="900" height="450" src="img/dependent_var.png">
</p>

There is a bit of a class imbalance in the dependent variable, `is_canceled`. I will take this into account when training my models by empirically choosing a balancing method (undersampling, oversampling, SMOTE, etc.) to apply to my data before training.