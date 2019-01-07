get_data.py
	from_2010(path, tag, train_hour, max_train_hour, pred_hour)
		to get raw feature e.g. temperature
		tag is the column in number

	get_time(path, max_hour, pred_hour)
		to return dade and time been normalized by 0~1
		date is divided by 366
		time is devided by 24

insert.py
	to fill in null features
	
train.py
	to train model
	the feature may be raw or been normalized
	
train_cross.py
	to do cross validation and testing for each year
	
train_PCA.py
	the feature been PCAed
	
tuning.py
	to tune parameters with raw feature or normalized

tuning_PCA.py
	to tune parameters with PCAed feature