using 
	CSV,
	DataFrames, 
	DecisionTree,
	Statistics, 
	Plots,
	Random

# training data read
building = CSV.file("data/building_metadata.csv") |> DataFrame!
train = CSV.file("data/train.csv") |> DataFrame!
weather = CSV.file("data/weather_train.csv") |> DataFrame!
full = join(train, building, on=:building_id, kind=:left) |> x -> join(x, weather, on=[:site_id, :timestamp], kind=:left)


# filling in missing values with mean
X = full[!, [:meter, :site_id, :square_feet, :year_built, :floor_count, :air_temperature, :cloud_coverage, :dew_temperature, :precip_depth_1_hr, :sea_level_pressure, :wind_direction, :wind_speed, :meter_reading]]
for col in names(X)[1:end-1]
    X[!, Symbol(col)] = recode(X[!, Symbol(col)], missing => mean(skipmissing(X[!, Symbol(col)])))
end


# train test split
function train_test_split(X::DataFrame, target_col::String, seed=0)
    function partitionTrainTest(data, at = 0.66, seed=0)
        if seed != 0
            Random.seed!(seed)
        end
        n = nrow(data)
        idx = shuffle(1:n)
        train_idx = view(idx, 1:floor(Int, at*n))
        test_idx = view(idx, (floor(Int, at*n)+1):n)
        data[train_idx,:], data[test_idx,:]
    end

    train,test = partitionTrainTest(X, 0.66, seed)
    X_train = select(train, Not(:meter_reading))
    y_train = train[!, Symbol(target_col)]
    X_test = select(test, Not(:meter_reading))
    y_test = test[!, Symbol(target_col)]
    return X_train, X_test, y_train, y_test
end

X_train, X_test, y_train, y_test = train_test_split(X, "meter_reading")


# simple decision tree
dt = DecisionTreeRegressor()
fit!(dt, convert(Matrix, X_train), convert(Array, y_train))
preds = predict(dt, convert(Matrix, X_test))

# scoring function
function logging(pred, target)
    return (log(pred + 1) - log(target + 1))^2
end
function result(pred, target)
    return sqrt(
        (1/length(pred)) * sum(logging.(pred, target))
    )
end
print("dt: " * String(result(preds, y_test)))


# random forest
n_subfeatures=-1; n_trees=20; partial_sampling=0.7; max_depth=-1
min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0

rf = build_forest(convert(Array, y_train), convert(Matrix, X_train),
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase)
preds = apply_forest(rf, convert(Matrix, X_test))
print("rf: " * results(preds, y_test))


# test data
test = CSV.file("data/test.csv") |> DataFrame!
weather_test = CSV.file("data/weather_test.csv") |> DataFrame!
full_test = join(test, building, on=:building_id, kind=:left) |> x -> join(x, weather_test, on=[:site_id, :timestamp], kind=:left)

X_final = full_test[!, [:meter, :site_id, :square_feet, :year_built, :floor_count, :air_temperature, :cloud_coverage, :dew_temperature, :precip_depth_1_hr, :sea_level_pressure, :wind_direction, :wind_speed]]
for col in names(X_final)
    X_final[!, Symbol(col)] = recode(X_final[!, Symbol(col)], missing => mean(skipmissing(X_final[!, Symbol(col)])))
end
final_preds = predict(dt, convert(Matrix, X_final))

# dt submission
X_final[!, :meter_reading] = final_preds
X_final[!, :row_id] = test.row_id
CSV.write("data/submission_dt.csv", X_final[!, [:row_id, :meter_reading]])

# rf submission
final_preds = apply_forest(rf, convert(Matrix, X_final))
X_final[!, :meter_reading] = final_preds
X_final[!, :row_id] = test.row_id
CSV.write("data/submission_rf.csv", X_final[!, [:row_id, :meter_reading]])

