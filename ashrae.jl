using
	CSV,
	DataFrames,
	Dates,
	DecisionTree,
	Distributed,
	StatsBase,
	Statistics,
	Plots,
	Random,
	XGBoost

# whether to reperform grid search
grid_search = false
# whether to work with only a 500k sample of the data
sample = false
# number of cores for distributed computing
addprocs(4)
# historical performance of algorithms
perf = CSV.file("data/performance.csv") |> DataFrame!


# training data read
building = CSV.file("data/building_metadata.csv") |> DataFrame!
train = CSV.file("data/train.csv") |> DataFrame!
if sample
	train = train[sample(axes(train, 1), 500000; replace = false, ordered = true), :]
end
weather = CSV.file("data/weather_train.csv") |> DataFrame!
full = join(train, building, on=:building_id, kind=:left) |> x -> join(x, weather, on=[:site_id, :timestamp], kind=:left)

# adding day, month, hour
date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:ss")
full.timestamp = Dates.DateTime.(full.timestamp, date_format)
full[!, :day] = Dates.day.(full.timestamp)
full[!, :month] = Dates.month.(full.timestamp)
full[!, :hour] = Dates.hour.(full.timestamp)
full[!, :dow] = Dates.dayofweek.(full.timestamp)


# test data read
test = CSV.file("data/test.csv") |> DataFrame!
weather_test = CSV.file("data/weather_test.csv") |> DataFrame!
full_test = join(test, building, on=:building_id, kind=:left) |> x -> join(x, weather_test, on=[:site_id, :timestamp], kind=:left)
# adding day, month, hour
date_format = Dates.DateFormat("yyyy-mm-dd HH:MM:ss")
full_test.timestamp = Dates.DateTime.(full_test.timestamp, date_format)
full_test[!, :day] = Dates.day.(full_test.timestamp)
full_test[!, :month] = Dates.month.(full_test.timestamp)
full_test[!, :hour] = Dates.hour.(full_test.timestamp)
full_test[!, :dow] = Dates.dayofweek.(full_test.timestamp)


### functions
# define one hot function
function one_hot(df, col)
	new_df = copy(df)
	possibilities = unique(df[!, col])
	names = string(col) .* "_" .* string.(possibilities)
	for i in 1:length(possibilities)
		new_df[!, Symbol(names[i])] .= 0
		new_df[new_df[!, col] .== possibilities[i], Symbol(names[i])] .= 1
	end
	select!(new_df, Not(col))
	return new_df
end

# scoring function
function logging(pred, target)
    return (log(pred + 1) - log(target + 1))^2
end
function result(pred, target)
    return sqrt(
        (1/length(pred)) * sum(logging.(pred, target))
    )
end

# generate the dataset needed for algorithms from "full" dfs
function gen_dataset(df, test_set=true)
	X = copy(df)
	# desired columns
	if test_set
		X = X[!, [:meter, :site_id, :square_feet, :year_built, :floor_count, :air_temperature, :cloud_coverage, :dew_temperature, :precip_depth_1_hr, :sea_level_pressure, :wind_direction, :wind_speed, :meter_reading, :day, :month, :hour, :dow]]
	else
		X = X[!, [:meter, :site_id, :square_feet, :year_built, :floor_count, :air_temperature, :cloud_coverage, :dew_temperature, :precip_depth_1_hr, :sea_level_pressure, :wind_direction, :wind_speed, :day, :month, :hour, :dow]]
	end
	# filling nas with means
	for col in names(X)[1:end-1]
	    X[!, Symbol(col)] = recode(X[!, Symbol(col)], missing => mean(skipmissing(X[!, Symbol(col)])))
	end
	# one hot encoding
	for col in [:meter, :site_id, :dow]
		X = one_hot(X, col)
	end
	return X
end

# train test split
function train_test_split(X::DataFrame, target_col::String, seed=0, train_share=0.8)
    function partitionTrainTest(data, at=0.8, seed=0)
        if seed != 0
            Random.seed!(seed)
        end
        n = nrow(data)
        idx = shuffle(1:n)
        train_idx = view(idx, 1:floor(Int, at*n))
        test_idx = view(idx, (floor(Int, at*n)+1):n)
        data[train_idx,:], data[test_idx,:]
    end

    train,test = partitionTrainTest(X, train_share, seed)
    X_train = select(train, Not(:meter_reading))
    y_train = train[!, Symbol(target_col)]
    X_test = select(test, Not(:meter_reading))
    y_test = test[!, Symbol(target_col)]
    return X_train, X_test, y_train, y_test
end
### functions


# training data
X = gen_dataset(full)
X_train, X_test, y_train, y_test = train_test_split(X, "meter_reading")
X_submission = gen_dataset(full_test, false)


# simple decision tree
if grid_search
	performance = []
	max_depth = [10, 50, 100, -1]
	min_samples_split = [2, 5, 20, 50]
	min_samples_leaf = [1, 5, 20, 50]
	min_purity_increase = [0.0, 0.001, 0.01]
	@distributed for md in max_depth
		for mss in min_samples_split
			for msl in min_samples_leaf
				for mpi in min_purity_increase
					params = "decision tree, max_depth:$md, min_samples_split:$mss, min_samples_leaf:$msl, min_purity_increase:$mpi"
					println(params)
					dt = DecisionTreeRegressor(
							max_depth = md,
							min_samples_split = mss,
							min_samples_leaf = msl,
							min_purity_increase = mpi
						)
					DecisionTree.fit!(dt, convert(Matrix, X_train), convert(Array, y_train))
					preds = DecisionTree.predict(dt, convert(Matrix, X_test))
					push!(performance, (
						params,
						result(preds, y_test),
						0,
						"NA"
					))
				end
			end
		end
	end
	tmp_perf = DataFrame(performance)
	names!(tmp_perf, Symbol.(["desc", "val_score", "test_score", "notes"]))
	perf = [perf; tmp_perf]
	CSV.write("data/performance.csv", perf)
	best_score = minimum([i[2] for i in performance])
	best_params = [i[1] for i in performance if i[2] == best_score]
	println(best_params)
end
# submission
dt = DecisionTreeRegressor(
	max_depth = 100,
	min_samples_split = 5,
	min_samples_leaf = 5,
	min_purity_increase = .001
)
DecisionTree.fit!(dt, convert(Matrix, select(X, Not(:meter_reading))), X[!, :meter_reading])
final_preds = DecisionTree.predict(dt, convert(Matrix, X_submission))
X_submission[!, :meter_reading] = final_preds
X_submission[!, :row_id] = test.row_id
CSV.write("data/submission_dt.csv", X_submission[!, [:row_id, :meter_reading]])


# random forest
n_subfeatures=-1; n_trees=20; partial_sampling=0.7; max_depth=-1
min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0
rf = @distributed build_forest(X[!, :meter_reading], convert(Matrix, select(X, Not(:meter_reading))),
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase)
#preds = apply_forest(rf, convert(Matrix, X_test))
#print("rf: " * results(preds, y_test))
# submission
final_preds = apply_forest(rf, convert(Matrix, X_submission))
X_submission[!, :meter_reading] = final_preds
X_submission[!, :row_id] = test.row_id
CSV.write("data/submission_rf.csv", X_submission[!, [:row_id, :meter_reading]])


# xgboost
bst = xgboost(select(X, Not(:meter_reading)), 2, label=X[!, :meter_reading], eta=1, max_depth=2)
# submission
final_preds = XGBoost.predict(bst, convert(Matrix, X_submission))
X_submission[!, :meter_reading] = final_preds
X_submission[!, :row_id] = test.row_id
CSV.write("data/submission_bst.csv", X_submission[!, [:row_id, :meter_reading]])
